# IQR: integer quantisation on the rotated int8 grid

IQ8R (`StillDeadcode/qwen3.8-27b-iq8r`) is Qwen3.8-27B with every trunk linear rotated by the
128-wide Hadamard (`transform=fwht128`), quantised to int8 with a bf16 scale per (row, 128 of K) by GPTQ
against the calibration Hessians, and served at `i8a8h`: the norm and gated producers write the
rotated int8 activation codes the GEMM reads (`qwen35_w8a8`). IQ6R .. IQ2R keep that recipe --
GPTQ, the rotation, integer codes, int8 activations -- and spend fewer bits on the weights.

## The formats

Every IQR weight is on IQ8R's grid: a value is an int8 times the scale of its group. A container
stores it narrower, and VRAM holds it at the same width: libr4d's IQR rows decode each code to its
int8 value where the weight is read (`r4d_iqr.h`; `gemm_iqr_nt_m64` up to 64 rows, `gemm_iqr_tiled`
for a prefill, `embed_lookup_iqr` and `logits_gemm_iqr` for the embedding and the head), with
rotated int8 activations on the `i8a8h` trunk:

| form | encoding | decoded as |
|---|---|---|
| signed | `iN*bf16[1x128]/fwht128`, N = 2..7 | sign extension |
| asymmetric | `uN*bf16[1x128]-uN[1x128]/fwht128` | code - zero |
| scalar codebook | `affine:codes=uN[1x1],scale=bf16[1x128],table=i8{1x2^N}/fwht128` | the entry |
| vector codebook | `affine:codes=uM[1x4],scale=bf16[1x128],grid=i8{E x 4}/fwht128` | the codeword's 4 values |

The embedding and the head decode the same forms, unrotated. The arithmetic is the int8 grid's
for every form: a narrower code changes the bytes read, not the values multiplied.

## The quantiser

`gptq` (libquant) over the rotated weight against the calibration Hessians:

- the codebooks are integer Lloyd tables (`data/codebooks/iqr-scalar.txt`) and k-means vector
  codebooks (`iqr-vq2-4x256.txt`, `iqr-vq3-4x4096.txt`), fitted to the rotated weights;
- `rule=search` scales; `dyn=1` chooses a group's scale when GPTQ reaches it;
- `cd=N` coordinate descent after the loop, per code or, for a vector grid, per codeword;
- the head is fitted by GPTQ too, against the head's input Hessian.

Measured on layer 3 (output error, relative): the scalar codebook beats a uniform grid by 12-25% at
two to five bits; the asymmetric grid wins at six. A 256 x 4 vector codebook with one descent pass
has 19% less error than the scalar 2-bit codebook with eight; the 4096 x 4 one 13% less than scalar
3-bit.

## Where the bits go

The KL a quantised band of layers contributes adds up across bands to within a few percent, so a
mix is chosen by a model: KL = floor + sum over bands of share x (KL of the band's grid - floor).
On the 4-bit trunk the eight bands of eight layers carry 9, 4, 8, 11, 13, 12, 16 and 28 % of the KL:
the last layers are the most sensitive, and the projections of a layer about equally so. The
embedding and the head are cheap to keep wide (1.27B parameters each, 0.047 bpw a bit) and costly
to narrow at the higher sizes: a 4-bit embedding by round to nearest costs IQ8R +0.00058 KL, a 6-bit
one +0.00003.

The 2-bit trunk's bands weigh the same way (12, 6, 7, 8, 10, 10, 17 and 31 %, measured by raising
one band of the 256 x 4 trunk to four bits). `rule=search` per group and a narrower scale group do
nothing at two bits (the vector codebook's error is the same at groups of 128 and 64): the bits
go to wider codebooks instead, and each half bit of a 4-wide codebook halves the error (256, 512,
1024 and 4096 codewords: 2.125 to 3.125 bits a weight with the scale).

## Results

Qwen3.8-27B against the competition's private tiers 1 + 2 (mean KL to the bf16 model's top-512,
bpw over the text model's 26.9B parameters, embedding and head included):

| recipe | bpw | KL mean | top-1 | the GGUF at that size | its bpw | its KL |
|---|---:|---:|---:|---|---:|---:|
| `q38-27b-iq6r-l` | 6.935 | 0.00092 | 98.91% | UD-Q6_K_L | 7.089 | 0.00129 |
| `q38-27b-iq6r` | 6.375 | 0.00123 | 98.73% | UD-Q6_K | 6.431 | 0.00194 |
| `q38-27b-iq6r-s` | 6.061 | 0.00151 | 98.48% | UD-Q5_K_XL | 6.102 | 0.00268 |
| `q38-27b-iq5r` | 5.420 | 0.00300 | 98.00% | UD-Q5_K_S | 5.444 | 0.00428 |
| `q38-27b-iq5r-s` | 5.075 | 0.00426 | 97.66% | UD-Q4_K_XL | 5.115 | 0.00608 |
| `q38-27b-iq4r-l` | 4.780 | 0.00609 | 97.19% | UD-Q4_K_M | 4.790 | 0.00820 |
| `q38-27b-iq4r` | 4.457 | 0.00904 | 96.56% | UD-Q4_K_S | 4.461 | 0.01057 |
| `q38-27b-iq4r-s` | 4.137 | 0.01327 | 95.50% | UD-IQ4_XS | 4.132 | 0.01499 |
| `q38-27b-iq3r-l` | 3.798 | 0.01887 | 94.78% | UD-Q3_K_XL | 3.803 | 0.02403 |
| `q38-27b-iq3r` | 3.459 | 0.03043 | 93.59% | UD-IQ3_S | 3.474 | 0.03473 |
| `q38-27b-iq3r-s` | 3.148 | 0.04469 | 92.29% | UD-IQ3_XXS | 3.145 | 0.05052 |
| `q38-27b-iq2r-l` | 2.790 | 0.06937 | 90.32% | UD-Q2_K_XL | 2.816 | 0.07920 |
| `q38-27b-iq2r` | 2.469 | 0.12234 | 87.02% | UD-IQ2_S | 2.487 | 0.12976 |

IQ6R is also below UD-Q6_K_L (7.089 bpw, 0.00129) and IQ5R below UD-Q5_K_M (5.773, 0.00328). By
source, IQ4R's lead over UD-Q4_K_S is widest on agent traces and reasoning (0.71 to 0.74 of its
KL; on the 256K agentic traces 0.00595 against 0.00822) and narrowest on chat, Wikipedia and long
fiction (wildchat 0.99, wikimedia 0.98, pg19 0.95).

A base convert (one grid over the whole trunk) is one GPTQ pass over the bf16 checkpoint, 35 to
100 minutes on 16 threads. A recipe converted from scratch costs one such pass; the mixes were
assembled from the bases instead (`rad-convert --reuse` with every base, which copies each weight
whose options match a base's), a few minutes each.

## IQR-V2: a trellis down each GPTQ column

IQR-V2 keeps the rotation, GPTQ, the int8 grid and the int8 activations, and changes what a code
is: the trunk's codes are a bitshift trellis (QTIP's) of 2^12 states run down each column of the
rotated weight (`tcq=12`). A row's value is the table entry of its state, the state being the last
12 bits of the column's codes; a run of 256 rows (`tcq_rows`) is tail-biting, so no bits go to a
start state. Down a column every row shares the column's H_ii, so the Viterbi's metric is GPTQ's
own objective for that column, and GPTQ's feedback to the later columns stays exact. The tables
are 4096 integers in -127..127 (`data/codebooks/iqr-tcq-L12-k2..k7.txt`), trained by Lloyd
iterations on Viterbi paths over the rotated weights; the scale is E4M3 per (row, 128) under one
f32 for the tensor (`scale=fp8_e4m3 scale2=f32 block2=*x*`), which the trellis was searched against
at the bf16 of their product. The GEMM walks each column's states in the tile and reads their
values from the table in LDS (`gemm_iqr`), so V2 serves on the `i8a8h` trunk at its codes' width.

On layer 3 the trellis has 21-29% less output error than V1's best grid at every rate from 2 to 6
bits. The mixes are chosen as V1's by band shares, over 24 units (8 bands x gate_up / down / the
attention or delta-net projections) and the k = 2 .. 7 trellises plus int8, by an exact Pareto
merge. The embedding is a rotated trellis too (unrotated at load); the head stays V1's scalar
GPTQ table, where a trellis clipped its outliers.

| recipe | bpw | KL mean | top-1 | V1 at that size | its bpw | its KL | KL |
|---|---:|---:|---:|---|---:|---:|---:|
| `q38-27b-iq8r-v2` | 8.040 | 0.00047 | 99.18% | IQ8R | 8.505 | 0.00046 | +2% |
| `q38-27b-iq7r-v2` | 7.418 | 0.00061 | 99.11% | -- | | | |
| `q38-27b-iq6r-l-v2` | 6.937 | 0.00073 | 99.05% | IQ6R-L | 6.935 | 0.00092 | -21% |
| `q38-27b-iq6r-v2` | 6.375 | 0.00094 | 98.92% | IQ6R | 6.375 | 0.00123 | -24% |
| `q38-27b-iq6r-s-v2` | 6.062 | 0.00115 | 98.78% | IQ6R-S | 6.061 | 0.00151 | -24% |
| `q38-27b-iq5r-v2` | 5.423 | 0.00204 | 98.34% | IQ5R | 5.420 | 0.00300 | -32% |
| `q38-27b-iq5r-s-v2` | 5.076 | 0.00273 | 98.07% | IQ5R-S | 5.075 | 0.00426 | -36% |
| `q38-27b-iq4r-l-v2` | 4.784 | 0.00377 | 97.78% | IQ4R-L | 4.780 | 0.00609 | -38% |
| `q38-27b-iq4r-v2` | 4.457 | 0.00562 | 97.29% | IQ4R | 4.457 | 0.00904 | -38% |
| `q38-27b-iq4r-s-v2` | 4.130 | 0.00832 | 96.67% | IQ4R-S | 4.137 | 0.01327 | -37% |
| `q38-27b-iq3r-l-v2` | 3.798 | 0.01198 | 96.04% | IQ3R-L | 3.798 | 0.01887 | -37% |
| `q38-27b-iq3r-v2` | 3.458 | 0.01880 | 95.06% | IQ3R | 3.459 | 0.03043 | -38% |
| `q38-27b-iq3r-s-v2` | 3.145 | 0.02798 | 93.75% | IQ3R-S | 3.148 | 0.04469 | -37% |
| `q38-27b-iq2r-l-v2` | 2.792 | 0.04698 | 92.03% | IQ2R-L | 2.790 | 0.06937 | -32% |
| `q38-27b-iq2r-v2` | 2.467 | 0.07303 | 90.15% | IQ2R | 2.469 | 0.12234 | -40% |

IQ4R-L-V2 is below the official FP8 checkpoint (8.766 bpw, 0.00389) at 55% of its size, and IQ7R-V2
is below UD-Q6_K_XL (7.417 bpw, 0.00114) at the same size. IQ8R-V2 holds IQ8R's KL in 5% fewer
bits.

## IQR-V3: a computed trellis under one scale a row

IQR-V3 keeps V2's tail-biting trellis inside GPTQ and changes the code and the scale. A state's
value is computed from the state instead of read from a table (`tcq_code=mad`): with
h = (st * A + B) mod 2^32 from the low 24 bits of each factor, the value is
bytesum(h & 0x3F3F3F3F) - 126, a near-Gaussian int8 (A = 0xB5297B, B = 0x2545F491, stored as the
`trellis.mad` plane beside the table they generate). A reader needs a multiply-add, a mask and one
byte-sum dot product a value and no table, so the trellis grows to 2^16 states (`tcq=16`; 2^15 at
two bits) at no decode cost. The scale is one f32 a row (`block=1x* scale=f32`), applied after the
K loop, with the codes searched at a gain over the row's absmax that rises with the width
(`tcq_gain`: 1.04 at two bits, 1.08, 1.12, 1.16, then 1.24 at six and seven), and higher for the
projections the attention heads feed (`attn_output`, `ssm_out`: 1.36, 1.5 and 1.6 at five, six and
seven bits). GPTQ's damping is 0.01.

The GEMM decodes in the tile (`gemm_iqr`). At one token the GEMV dots each value's hash with its
activation byte repeated in all four bytes, so a 4-bit `ffn_gate_up` takes 0.58 of FP8's time
(V2 0.80), and a decoder layer's trunk GEMMs at 2048 tokens take 1.00-1.06x FP8's (V2
1.19-1.21x), since one scale a row folds as FP8's block scale does. The mixes are chosen as V2's,
from the k = 2 .. 7 bases and IQ8R's int8 grid at V2's sizes; IQ8R-V3 is IQ8R-V2's container, the
int8 grid both share.

| recipe | bpw | KL mean | top-1 | V2 at that size | its bpw | its KL | KL |
|---|---:|---:|---:|---|---:|---:|---:|
| `q38-27b-iq8r-v3` | 8.040 | 0.00047 | 99.18% | IQ8R-V2 | 8.040 | 0.00047 | 0% |
| `q38-27b-iq7r-v3` | 7.420 | 0.00060 | 99.12% | IQ7R-V2 | 7.418 | 0.00061 | -2% |
| `q38-27b-iq6r-l-v3` | 6.939 | 0.00072 | 99.04% | IQ6R-L-V2 | 6.937 | 0.00073 | -2% |
| `q38-27b-iq6r-v3` | 6.372 | 0.00088 | 98.93% | IQ6R-V2 | 6.375 | 0.00094 | -6% |
| `q38-27b-iq6r-s-v3` | 6.049 | 0.00106 | 98.83% | IQ6R-S-V2 | 6.062 | 0.00115 | -8% |
| `q38-27b-iq5r-v3` | 5.422 | 0.00182 | 98.38% | IQ5R-V2 | 5.423 | 0.00204 | -11% |
| `q38-27b-iq5r-s-v3` | 5.064 | 0.00249 | 98.15% | IQ5R-S-V2 | 5.076 | 0.00273 | -9% |
| `q38-27b-iq4r-l-v3` | 4.783 | 0.00342 | 97.86% | IQ4R-L-V2 | 4.784 | 0.00377 | -9% |
| `q38-27b-iq4r-v3` | 4.458 | 0.00512 | 97.45% | IQ4R-V2 | 4.457 | 0.00562 | -9% |
| `q38-27b-iq4r-s-v3` | 4.131 | 0.00762 | 96.90% | IQ4R-S-V2 | 4.130 | 0.00832 | -8% |
| `q38-27b-iq3r-l-v3` | 3.798 | 0.01129 | 96.16% | IQ3R-L-V2 | 3.798 | 0.01198 | -6% |
| `q38-27b-iq3r-v3` | 3.459 | 0.01698 | 95.35% | IQ3R-V2 | 3.458 | 0.01880 | -10% |
| `q38-27b-iq3r-s-v3` | 3.145 | 0.02584 | 94.02% | IQ3R-S-V2 | 3.145 | 0.02798 | -8% |
| `q38-27b-iq2r-l-v3` | 2.786 | 0.04241 | 92.62% | IQ2R-L-V2 | 2.792 | 0.04698 | -10% |
| `q38-27b-iq2r-v3` | 2.465 | 0.06807 | 90.54% | IQ2R-V2 | 2.467 | 0.07303 | -7% |

The design, the per-weight and GEMM measurements and the comparison with V1 and V2 are in the
IQR-V3 technical report (radiance.stilldeadco.de/whitepapers/iqr-v3.pdf).

## IQR-V4: measured widths, a vocabulary-weighted head and faster kernels

IQR-V4 keeps V3's computed trellis and its one f32 scale a row, and changes how the bits are spent and
how the GEMMs read them. GPTQ runs in act order inside the column trellis (`tcq_order=1`: the columns
are fitted in descending Hessian diagonal and the codes stored in column order), and the FFN weights'
trellis runs grow to 512 or 1024 rows (`tcq_rows`). The projections the attention heads feed
(`attn_output`, `ssm_out`) take a bf16 scale a (row, 128) block from five bits up (`group=128
scale=bf16 dyn=1`), where V3's code reached its resolution first. The widths come from measured
prices: each band of eight layers and projection is priced by the KL a width step costs it, measured
at the 2/3-, 3/4- and 4/5-bit boundaries and shrunk halfway to the model's prices, and layer 63's
`ffn_gate_up`, the one tensor whose price is several times its band's, sits one width up (the V4S
entries, to IQ3R-L); from IQ4R-S up the model's prices are as good and are kept (V4W).

The vocabulary edges carry about a tenth of every entry's bytes. The lm_head's KL weighs each
vocabulary row by its token's probability, so below IQ6R-L the head is vocabulary-weighted: its rows
are four width classes of the computed trellis (`output.k<B>`, a 256-row run a class, chosen by the
row's probability-weighted output error at a 4.0, 4.5 or 5.0 b mean; `data/vhead/`, `tools/vhead.py`),
decoded in one launch by `logits_gemm_vocab` with the head input held at 16 bits and the logits
gathered back into vocabulary order. At 1 to 16 tokens it is faster than the int8 head it replaces.
IQ6R-L-V4 to IQ8R-V4 keep the int8 head, now GPTQ'd against the head's input Hessian, which costs no
bytes and lowers the median KL about 10%.

The kernels: the one-token GEMV gives each 16-row tile a block whose waves split K, so a GEMM's tiles
finish together instead of trailing off on a few long waves; a 16-token tile without block barriers serves 2 to 16 tokens; and
the activation quantisers after `gate_up`, the delta net's decode fold and the attention gate are folded
into the kernels that produce their input. On Qwen3.8-27B at TP1 IQ4R-S-V4 decodes one stream 4% faster
than the MXFP4 checkpoint at 5x lower KL, and IQ3R-S-V4 is faster at every concurrency.

| recipe | bpw | KL mean | top-1 | V3 at that size | its bpw | its KL | KL |
|---|---:|---:|---:|---|---:|---:|---:|
| `q38-27b-iq8r-v4` | 8.040 | 0.00046 | 99.22% | IQ8R-V3 | 8.040 | 0.00047 | -2% |
| `q38-27b-iq7r-v4` | 7.420 | 0.00059 | 99.15% | IQ7R-V3 | 7.420 | 0.00060 | -2% |
| `q38-27b-iq6r-l-v4` | 6.939 | 0.00068 | 99.08% | IQ6R-L-V3 | 6.939 | 0.00072 | -6% |
| `q38-27b-iq6r-v4` | 6.439 | 0.00081 | 98.96% | IQ6R-V3 | 6.372 | 0.00088 | -8% |
| `q38-27b-iq6r-s-v4` | 6.063 | 0.00100 | 98.86% | IQ6R-S-V3 | 6.049 | 0.00106 | -6% |
| `q38-27b-iq5r-v4` | 5.433 | 0.00154 | 98.60% | IQ5R-V3 | 5.422 | 0.00182 | -15% |
| `q38-27b-iq5r-s-v4` | 5.071 | 0.00223 | 98.33% | IQ5R-S-V3 | 5.064 | 0.00249 | -10% |
| `q38-27b-iq4r-l-v4` | 4.793 | 0.00297 | 98.07% | IQ4R-L-V3 | 4.783 | 0.00342 | -13% |
| `q38-27b-iq4r-v4` | 4.473 | 0.00444 | 97.69% | IQ4R-V3 | 4.458 | 0.00512 | -13% |
| `q38-27b-iq4r-s-v4` | 4.142 | 0.00643 | 97.12% | IQ4R-S-V3 | 4.131 | 0.00762 | -16% |
| `q38-27b-iq3r-l-v4` | 3.804 | 0.00917 | 96.61% | IQ3R-L-V3 | 3.798 | 0.01129 | -19% |
| `q38-27b-iq3r-v4` | 3.472 | 0.01495 | 95.74% | IQ3R-V3 | 3.459 | 0.01698 | -12% |
| `q38-27b-iq3r-s-v4` | 3.157 | 0.02241 | 94.75% | IQ3R-S-V3 | 3.145 | 0.02584 | -13% |
| `q38-27b-iq2r-l-v4` | 2.797 | 0.03602 | 93.29% | IQ2R-L-V3 | 2.786 | 0.04241 | -15% |
| `q38-27b-iq2r-v4` | 2.470 | 0.06113 | 91.23% | IQ2R-V3 | 2.465 | 0.06807 | -10% |

IQ4R-L-V4 is 24% below the official FP8 checkpoint's KL (8.766 bpw, 0.00389) at 55% of its size, and
IQ2R-L-V4, at 2.8 bpw, is below the AWQ-tuned MXFP4 checkpoint (4.994 bpw, 0.03824). The design, the
kernels and the measurements are in the IQR-V4 technical report
(radiance.stilldeadco.de/whitepapers/iqr-v4.pdf).
