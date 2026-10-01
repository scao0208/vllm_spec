"""CPU regressions for PR #2 (requires PyTorch, not vLLM/Triton/a GPU).

The scripts initialize a GPU driver/model stack at import time. Compile their
actual host definitions from AST so these tests can exercise them on CPU.
Only kernel launches and the paged-attention dependency are replaced; these
tests do NOT validate Triton code generation or GPU numerical correctness.

Run: python -m unittest discover -s tests -p 'test_eagle_tree_regressions.py' -v
"""

import ast
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_definitions(filename, names, **namespace):
    """Load unmodified bodies; strip only top-level reporting decorators."""
    path = SCRIPTS / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    for name in names:
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                    and n.name == name)
        node.decorator_list = []
        nodes.append(node)
    namespace.update(torch=torch, F=F, nullcontext=nullcontext,
                     __name__=__name__)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace


class DeepTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trees = runpy.run_path(str(SCRIPTS / "eagle_tree_choices.py"))

    def assert_valid_tree(self, tree, budget, expected_depth):
        self.assertLessEqual(len(tree), budget)
        self.assertEqual(len(set(tree)), len(tree))
        self.assertEqual(tree, sorted(tree, key=lambda p: (len(p), p)))
        self.assertEqual(max(map(len, tree), default=0), expected_depth)
        positions = {node: i for i, node in enumerate(tree)}
        for node, i in positions.items():
            for depth in range(1, len(node)):
                self.assertLess(positions[node[:depth]], i)
        for depth in range(1, expected_depth + 1):
            self.assertIn((0,) * depth, positions)

    def test_presets_keep_requested_depth_and_size(self):
        for budget, depth in [(64, 8), (256, 12), (512, 14), (1024, 16)]:
            with self.subTest(budget=budget):
                tree = self.trees[f"eagle_tree_{budget}"]
                self.assertEqual(len(tree), budget)
                self.assert_valid_tree(tree, budget, depth)

    def test_small_budgets_remain_prefix_closed(self):
        generate = self.trees["generate_deep_eagle_tree"]
        for top_k in [1, 2, 8]:
            for depth in [0, 1, 5, 12]:
                for budget in [0, 1, 3, 5, 12, 20]:
                    with self.subTest(top_k=top_k, depth=depth, budget=budget):
                        tree = generate(top_k, depth, budget)
                        self.assert_valid_tree(tree, budget, min(depth, budget))
                        self.assertEqual(tree, generate(top_k, depth, budget))

    def test_invalid_parameters(self):
        for args in [(0, 4, 10), (4, -1, 10), (4, 4, -1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.trees["generate_deep_eagle_tree"](*args)


class MixedDecodeTests(unittest.TestCase):
    def test_all_decode_variants_write_single_and_tree_rows(self):
        torch.manual_seed(7)
        heads, kv_heads, dim, block_size = 4, 2, 32, 4
        q_lengths = [1, 3, 0, 1, 2]
        starts = torch.tensor([0, 1, 4, 4, 5, 7], dtype=torch.int32)
        lengths = torch.tensor([5, 7, 0, 6, 4], dtype=torch.int32)
        blocks = torch.tensor([[8, 3], [1, 10], [2, 4], [11, 5], [0, 9]])
        query = torch.randn(7, heads, dim, dtype=torch.float16)
        key = torch.randn(12, block_size, kv_heads, dim, dtype=torch.float16)
        value = torch.randn_like(key)
        # A branching tree: the last two queries cannot see each other.
        bias = torch.tensor([[0., -torch.inf, -torch.inf],
                             [0., 0., -torch.inf], [0., -torch.inf, 0.]])
        meta = SimpleNamespace(query_start_loc=starts, seq_lens=lengths,
                               block_table=blocks, tree_attn_bias=bias)
        impl = SimpleNamespace(num_heads=heads, num_kv_heads=kv_heads,
                               head_size=dim, scale=dim ** -0.5,
                               alibi_slopes=None, sliding_window=(-1, -1),
                               logits_soft_cap=0.)
        layer = SimpleNamespace(_k_scale=torch.tensor(1.),
                                _v_scale=torch.tensor(1.))

        def gather(cache, table, length):
            flat = cache[table].reshape(-1, kv_heads, dim)[:length]
            return flat.transpose(0, 1).repeat_interleave(heads // kv_heads, 0)

        # Independent dense oracle over each request's own paged KV.
        expected = []
        for i, q_len in enumerate(q_lengths):
            if not q_len:
                continue
            start, length = int(starts[i]), int(lengths[i])
            q = query[start:start + q_len].transpose(0, 1).double()
            k = gather(key, blocks[i], length).double()
            v = gather(value, blocks[i], length).double()
            scores = q @ k.transpose(-1, -2) * impl.scale
            if q_len > 1:
                scores[..., -q_len:] += bias[:q_len, :q_len]
            result = (scores.softmax(-1) @ v).transpose(0, 1)
            expected.append(result.reshape(q_len, -1).half())
        expected = torch.cat(expected)

        for name in ["_eagle_tree_decode", "_sparse_tree_decode",
                     "_bitmap_tree_decode"]:
            with self.subTest(variant=name):
                fallback_rows, tree_lengths = [], []

                def unified_attention(**kwargs):
                    row = len(fallback_rows)
                    seq_idx = [0, 3][row]
                    fallback_rows.append(seq_idx)
                    self.assertEqual(kwargs["cu_seqlens_q"].tolist(), [0, 1])
                    self.assertEqual(kwargs["max_seqlen_q"], 1)
                    self.assertEqual(kwargs["seqused_k"].tolist(),
                                     [int(lengths[seq_idx])])
                    self.assertEqual(kwargs["max_seqlen_k"], int(lengths[seq_idx]))
                    torch.testing.assert_close(kwargs["block_table"],
                                               blocks[seq_idx:seq_idx + 1])
                    self.assertTrue(kwargs["causal"])
                    self.assertEqual(kwargs["window_size"], impl.sliding_window)
                    self.assertEqual(kwargs["softcap"], impl.logits_soft_cap)
                    self.assertEqual(kwargs["k_descale"].shape, (1, kv_heads))
                    self.assertEqual(kwargs["v_descale"].shape, (1, kv_heads))
                    self.assertIsNone(kwargs["q_descale"])
                    k = gather(kwargs["k"], kwargs["block_table"][0],
                               kwargs["max_seqlen_k"])
                    v = gather(kwargs["v"], kwargs["block_table"][0],
                               kwargs["max_seqlen_k"])
                    result = F.scaled_dot_product_attention(
                        kwargs["q"].transpose(0, 1), k, v,
                        scale=kwargs["softmax_scale"])
                    kwargs["out"].copy_(result.transpose(0, 1).reshape(1, -1))

                def tree_kernel(q, k, v, mask, scale, *unused):
                    n = q.shape[2]
                    tree_lengths.append(n)
                    full_mask = F.pad(mask[..., :n, :n],
                                      (k.shape[2] - n, 0), value=0)
                    return F.scaled_dot_product_attention(q, k, v,
                                                         attn_mask=full_mask,
                                                         scale=scale)

                kernel = SimpleNamespace(apply=tree_kernel)
                tensors = dict(tree_mask_4d=bias[None, None].half(),
                               block_indices=None, block_counts=None, bitmaps=None)
                ns = load_definitions(
                    "benchmark_eagle_tree.py", ["_single_token_decode", name],
                    _get_device_tensors=lambda device: tensors,
                    _attention_sparse=kernel, _attention_bitmap=kernel,
                    MAX_SPARSE_BLOCKS=1, W=1)
                module = ModuleType("vllm.v1.attention.ops.triton_unified_attention")
                module.unified_attention = unified_attention
                output = torch.full((9, heads * dim), torch.nan, dtype=query.dtype)
                with patch.dict(sys.modules, {module.__name__: module}):
                    args = (impl, layer, query, key, value, output, meta)
                    ns[name](*args, *([kernel] if name == "_eagle_tree_decode" else []))
                self.assertEqual(fallback_rows, [0, 3])
                self.assertEqual(tree_lengths, [3, 2])
                torch.testing.assert_close(output[:7], expected, atol=2e-3, rtol=2e-3)
                self.assertTrue(torch.isnan(output[7:]).all())


class BenchmarkTests(unittest.TestCase):
    def benchmark(self, attention, do_bench, flash=None):
        ns = load_definitions(
            "eagle_tree_attention.py", ["bench_flash_attention"], DEVICE="cpu",
            attention=attention, flash_attn_func=flash,
            triton=SimpleNamespace(testing=SimpleNamespace(do_bench=do_bench)))
        return ns["bench_flash_attention"]

    def test_forward_benchmark_causal_and_noncausal(self):
        for causal in [True, False]:
            with self.subTest(causal=causal):
                calls = []

                def attention(q, k, v, mask, scale, warp_specialize):
                    self.assertFalse(q.requires_grad or k.requires_grad or v.requires_grad)
                    self.assertEqual(scale, 1.3)
                    if causal:
                        self.assertEqual(mask.shape, (1, 1, 8, 8))
                        self.assertEqual(mask.dtype, q.dtype)
                        self.assertEqual(mask.device, q.device)
                        lower = torch.ones(8, 8, dtype=torch.bool).tril()
                        self.assertTrue((mask[0, 0][lower] == 0).all())
                        self.assertTrue(torch.isneginf(mask[0, 0][~lower]).all())
                    else:
                        self.assertIsNone(mask)
                    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
                    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
                    torch.testing.assert_close(result, ref)
                    calls.append(mask)
                    return result

                def timing(fn):
                    fn()
                    fn()
                    return 1.

                result = self.benchmark(attention, timing)(
                    2, 3, 8, 32, causal, False, "fwd", "triton-fp16")
                self.assertGreater(result, 0)
                self.assertEqual(len(calls), 2)
                self.assertIs(calls[0], calls[1])  # Mask allocation is outside timing.

    def test_triton_backward_rejected_before_allocation_or_timing(self):
        attention, timing = Mock(), Mock()
        bench = self.benchmark(attention, timing)
        with patch.object(torch, "randn") as randn:
            with self.assertRaisesRegex(NotImplementedError, "forward-only"):
                bench(1, 1, 8, 32, True, False, "bwd", "triton-fp16")
            randn.assert_not_called()
        attention.assert_not_called()
        timing.assert_not_called()

    def test_flash_backward_dispatch_remains_differentiable(self):
        # Real CPU autograd, substituting SDPA for the optional Flash kernel.
        inputs = []

        def flash(qkv, causal):
            inputs.append(qkv)
            q, k, v = qkv.unbind(2)
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=causal).transpose(1, 2)

        def timing(fn):
            fn()
            return 1.

        result = self.benchmark(None, timing, flash)(
            1, 2, 8, 32, True, False, "bwd", "flash")
        self.assertGreater(result, 0)
        self.assertIsNotNone(inputs[0].grad)
        self.assertTrue(torch.isfinite(inputs[0].grad).all())


class DeviceAndAutogradTests(unittest.TestCase):
    def wrappers(self, launch=None):
        class Kernel:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    if launch is not None:
                        launch()
                    # Initialize output; GPU arithmetic is outside this test.
                    args[7 if "FP8_OUTPUT" in kwargs else 3].zero_()
                return run

        return load_definitions(
            "eagle_tree_attention.py", ["_device_context", "_attention", "_attention_strided"],
            triton=SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                   set_allocator=lambda fn: None),
            is_hip=lambda: False, is_hopper=lambda: False,
            is_blackwell=lambda: False, supports_host_descriptor=lambda: False,
            _attn_fwd=Kernel(), _attn_fwd_strided=Kernel())

    def test_cpu_forwards_do_not_touch_cuda(self):
        ns = self.wrappers()
        with ExitStack() as stack:
            for name in ["device", "current_device", "set_device", "get_device_capability"]:
                stack.enter_context(patch.object(torch.cuda, name,
                                                side_effect=AssertionError(name)))
            for name in ["_attention", "_attention_strided"]:
                with self.subTest(wrapper=name):
                    q = torch.randn(1, 2, 8, 32)
                    out = ns[name].apply(q, q, q, None, 0.5)
                    self.assertEqual(out.shape, q.shape)
                    self.assertTrue((out == 0).all())

    def test_other_device_contexts_do_not_touch_cuda(self):
        ns = self.wrappers()
        with patch.object(torch.cuda, "device", side_effect=AssertionError("CUDA")):
            for device in ["cpu", "xpu:0", "mps:0"]:
                with ns["_device_context"](torch.device(device)):
                    pass

    def test_autograd_backward_has_explicit_error(self):
        ns = self.wrappers()
        for name in ["_attention", "_attention_strided"]:
            with self.subTest(wrapper=name):
                q = torch.randn(1, 1, 8, 32, requires_grad=True)
                out = ns[name].apply(q, q, q, None, 0.5)
                with self.assertRaisesRegex(NotImplementedError, "forward-only"):
                    out.sum().backward()

    def test_cuda_context_restores_after_success_and_launch_failure(self):
        ns = self.wrappers()
        for fail in [False, True]:
            events = []

            @contextmanager
            def device_context(device):
                events.append(("enter", device))
                try:
                    yield
                finally:
                    events.append(("restore", device))

            with patch.object(torch.cuda, "device", side_effect=device_context):
                try:
                    with ns["_device_context"](torch.device("cuda:2")):
                        if fail:
                            raise RuntimeError("launch failed")
                except RuntimeError:
                    self.assertTrue(fail)
            self.assertEqual(events, [("enter", torch.device("cuda:2")),
                                      ("restore", torch.device("cuda:2"))])

    def test_both_forwards_unwind_context_on_launch_failure(self):
        for name in ["_attention", "_attention_strided"]:
            with self.subTest(wrapper=name):
                events = []

                @contextmanager
                def context(device):
                    events.append("enter")
                    try:
                        yield
                    finally:
                        events.append("restore")

                def launch():
                    self.assertEqual(events, ["enter"])
                    raise RuntimeError("launch failed")

                ns = self.wrappers(launch)
                ns["_device_context"] = context
                q = torch.randn(1, 1, 8, 32)
                with self.assertRaisesRegex(RuntimeError, "launch failed"):
                    ns[name].apply(q, q, q, None, 0.5)
                self.assertEqual(events, ["enter", "restore"])


if __name__ == "__main__":
    unittest.main()
