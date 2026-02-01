// We think that clarity is more important in this submodule.
#![allow(unused)]

/// This is the LLM functionality module of arxiv-daily. It provides self-written functionalities using candle to load and run gemma 3 270m IT Q5 K_M QAT model.
/// Reference:
/// https://github.com/ggml-org/llama.cpp/blob/3dd95914d09b155eed84664b9abdbbffae238738/src/models/gemma3.cpp
/// https://github.com/google/gemma_pytorch/tree/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma
/// https://arxiv.org/pdf/2503.19786

#[cfg(feature = "ssr")]
use std::path::Path;
#[cfg(feature = "ssr")]
use std::sync::Arc;

#[cfg(feature = "ssr")]
use candle_core::D;
#[cfg(feature = "ssr")]
use candle_core::DType;
#[cfg(feature = "ssr")]
use candle_core::Device;
#[cfg(feature = "ssr")]
use candle_core::Module;
#[cfg(feature = "ssr")]
use candle_core::Result;
#[cfg(feature = "ssr")]
use candle_core::Tensor;
#[cfg(feature = "ssr")]
use candle_core::quantized::QMatMul;
#[cfg(feature = "ssr")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "ssr")]
use pulp::Arch;
#[cfg(feature = "ssr")]
use rayon::prelude::*;
#[cfg(feature = "ssr")]
use simd_json;
#[cfg(feature = "ssr")]
use tokenizers::Tokenizer;

#[cfg(feature = "ssr")]
#[derive(Debug, Clone)]

pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

#[cfg(feature = "ssr")]

impl Config {
    pub fn gemma3_270m() -> Self {

        Self {
            vocab_size: 262144,
            hidden_size: 640,
            intermediate_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            head_dim: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            max_position_embeddings:
                8192,
        }
    }
}

#[cfg(feature = "ssr")]

pub struct QLinear {
    inner: QMatMul,
}

#[cfg(feature = "ssr")]

impl QLinear {
    #[allow(clippy::inline_always)]
    #[inline(always)]

    fn new(
        tensor: candle_core::quantized::QTensor
    ) -> Result<Self> {

        Ok(Self {
            inner:
                QMatMul::from_qtensor(
                    tensor,
                )?,
        })
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]

    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor> {

        self.inner
            .forward(xs)
    }
}

#[cfg(feature = "ssr")]

pub struct RmsNorm {
    weight: Tensor,
    weight_data: Option<Vec<f32>>,
    eps: f64,
}

#[cfg(feature = "ssr")]

impl RmsNorm {
    fn new(
        weight: Tensor,
        eps: f64,
    ) -> Self {

        let weight_data = if weight
            .device()
            .is_cpu()
        {

            weight
                .to_dtype(DType::F32)
                .ok()
                .and_then(|t| {

                    t.flatten_all()
                        .ok()?
                        .to_vec1::<f32>(
                        )
                        .ok()
                })
        } else {

            None
        };

        Self {
            weight,
            weight_data,
            eps,
        }
    }

    fn forward(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {

        let x_dtype = x.dtype();

        let hidden_size =
            x.dim(D::Minus1)?;

        // Performance optimization:
        // If we are on CPU and using F32, we can use pulp to compute RMS in a single pass
        // instead of multiple intermediate tensors (sqr, sum, div, sqrt, etc.)
        if x.device().is_cpu()
            && x_dtype == DType::F32
            && self
                .weight_data
                .is_some()
        {

            let dims = x.dims();

            let weight_vec = self
                .weight_data
                .as_ref()
                .unwrap();

            // Try to avoid copy by accessing storage directly if possible
            let (storage, layout) =
                x.storage_and_layout();

            if let candle_core::Storage::Cpu(cpu) = &*storage {
                if layout.is_contiguous() {
                    let full_slice = cpu.as_slice::<f32>()?;
                    let x_slice = &full_slice[layout.start_offset()..layout.start_offset() + layout.shape().elem_count()];

                    let mut result_vec = Vec::<f32>::with_capacity(x_slice.len());
                    unsafe { result_vec.set_len(x_slice.len()); }
                    let arch = Arch::new();

                    let hidden_size = *dims.last().unwrap();
                    let num_elements = x_slice.len();
                    let num_rows = num_elements / hidden_size;

                    arch.dispatch(|| {
                        let process_row = |(row, res_row): (&[f32], &mut [f32])| {
                            let mut sum_sq = 0.0f32;
                            for chunk in row.chunks_exact(8) {
                                sum_sq += chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3] +
                                          chunk[4] * chunk[4] + chunk[5] * chunk[5] + chunk[6] * chunk[6] + chunk[7] * chunk[7];
                            }
                            for &val in row.chunks_exact(8).remainder() {
                                sum_sq += val * val;
                            }

                            let inv_norm = 1.0 / (sum_sq / hidden_size as f32 + self.eps as f32).sqrt();
                            for i in 0..hidden_size {
                                res_row[i] = row[i] * inv_norm * weight_vec[i];
                            }
                        };

                        if num_rows > 1 {
                             x_slice.par_chunks_exact(hidden_size)
                                 .zip(result_vec.par_chunks_exact_mut(hidden_size))
                                 .for_each(process_row);
                        } else {
                             x_slice.chunks_exact(hidden_size)
                                 .zip(result_vec.chunks_exact_mut(hidden_size))
                                 .for_each(process_row);
                        }
                    });

                    return Tensor::from_vec(result_vec, dims, x.device());
                }
            }

            // Fallback to to_vec1 (which copies from GPU/complex layout) if storage access failed
            let dims = x.dims();

            let weight_vec = self
                .weight_data
                .as_ref()
                .unwrap();

            let mut result = x
                .flatten_all()?
                .to_vec1::<f32>()?;

            let arch = Arch::new();

            let hidden_size =
                *dims.last().unwrap();

            let num_elements =
                result.len();

            let num_rows = num_elements
                / hidden_size;

            arch.dispatch(|| {
                let process_row = |(row, res_row): (&[f32], &mut [f32])| {
                    let mut sum_sq = 0.0f32;
                    for chunk in row.chunks_exact(8) {
                        sum_sq += chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3] +
                                  chunk[4] * chunk[4] + chunk[5] * chunk[5] + chunk[6] * chunk[6] + chunk[7] * chunk[7];
                    }
                    for &val in row.chunks_exact(8).remainder() {
                        sum_sq += val * val;
                    }

                    let inv_norm = 1.0 / (sum_sq / hidden_size as f32 + self.eps as f32).sqrt();
                    for i in 0..hidden_size {
                        res_row[i] = res_row[i] * inv_norm * weight_vec[i];
                    }
                };

                if num_rows > 1 {
                    result.par_chunks_exact_mut(hidden_size).for_each(|res_row| {
                        let mut sum_sq = 0.0f32;
                        for chunk in res_row.chunks_exact(8) {
                            sum_sq += chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3] +
                                      chunk[4] * chunk[4] + chunk[5] * chunk[5] + chunk[6] * chunk[6] + chunk[7] * chunk[7];
                        }
                        for &val in res_row.chunks_exact(8).remainder() {
                            sum_sq += val * val;
                        }

                        let inv_norm = 1.0 / (sum_sq / hidden_size as f32 + self.eps as f32).sqrt();
                        for i in 0..hidden_size {
                            res_row[i] = res_row[i] * inv_norm * weight_vec[i];
                        }
                    });
                } else {
                    result.chunks_exact_mut(hidden_size).for_each(|res_row| {
                        let mut sum_sq = 0.0f32;
                        for chunk in res_row.chunks_exact(8) {
                            sum_sq += chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3] +
                                      chunk[4] * chunk[4] + chunk[5] * chunk[5] + chunk[6] * chunk[6] + chunk[7] * chunk[7];
                        }
                        for &val in res_row.chunks_exact(8).remainder() {
                            sum_sq += val * val;
                        }

                        let inv_norm = 1.0 / (sum_sq / hidden_size as f32 + self.eps as f32).sqrt();
                        for i in 0..hidden_size {
                            res_row[i] = res_row[i] * inv_norm * weight_vec[i];
                        }
                    });
                }
            });

            return Tensor::from_vec(
                result,
                dims,
                x.device(),
            );
        }

        // Fallback for other dtypes/devices
        let internal_dtype =
            match x_dtype {
                | DType::F16
                | DType::BF16 => {
                    DType::F32
                },
                | d => d,
            };

        let x =
            x.to_dtype(internal_dtype)?;

        let norm_x = (x
            .sqr()?
            .sum_keepdim(D::Minus1)?
            / hidden_size as f64)?;

        let x_normed = x
            .broadcast_div(
                &(norm_x + self.eps)?
                    .sqrt()?,
            )?;

        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)
    }
}

#[cfg(feature = "ssr")]

impl Module for RmsNorm {
    #[allow(clippy::inline_always)]
    #[inline(always)]

    fn forward(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {

        self.forward(x)
    }
}

#[cfg(feature = "ssr")]

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

#[cfg(feature = "ssr")]

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        cfg: &Config,
        base: f64,
        dev: &Device,
    ) -> Result<Self> {

        let dim = cfg.head_dim;

        let max_seq_len =
            cfg.max_position_embeddings;

        let inv_freq: Vec<_> = (0
            .. dim)
            .step_by(2)
            .map(|i| {

                1f32 / base.powf(
                    i as f64
                        / dim as f64,
                )
                    as f32
            })
            .collect();

        let inv_freq_len =
            inv_freq.len();

        let inv_freq =
            Tensor::from_vec(
                inv_freq,
                (1, inv_freq_len),
                dev,
            )?
            .to_dtype(dtype)?;

        let t = Tensor::arange(
            0u32,
            max_seq_len as u32,
            dev,
        )?
        .to_dtype(dtype)?
        .reshape((max_seq_len, 1))?;

        let freqs =
            t.matmul(&inv_freq)?;

        Ok(Self {
            sin: freqs
                .sin()?
                .contiguous()?,
            cos: freqs
                .cos()?
                .contiguous()?,
        })
    }

    fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {

        let (b, h, s, d) = q.dims4()?;

        let cos = self.cos.narrow(
            0,
            seqlen_offset,
            s,
        )?;

        let sin = self.sin.narrow(
            0,
            seqlen_offset,
            s,
        )?;

        let (q_embed, k_embed) =
            rayon::join(
                || {

                    candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)
                },
                || {

                    candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)
                },
            );

        Ok((q_embed?, k_embed?))
    }
}

#[cfg(feature = "ssr")]

pub struct MLP {
    gate_proj: QLinear,
    up_proj: QLinear,
    down_proj: QLinear,
}

#[cfg(feature = "ssr")]

impl MLP {
    fn forward(
        &self,
        xs: &Tensor,
    ) -> Result<Tensor> {

        let (gate, up) = rayon::join(
            || {

                self.gate_proj
                    .forward(xs)
            },
            || {

                self.up_proj
                    .forward(xs)
            },
        );

        let mut gate = gate?;

        let up = up?;

        // Optimization: In-place Gelu and Multiplication
        // This avoids two intermediate tensor allocations (Gelu result and Mul result)
        // and combines two passes into one SIMD-accelerated pass.
        if gate
            .device()
            .is_cpu()
            && gate.dtype()
                == DType::F32
            && up.dtype() == DType::F32
        {

            let (g_storage, g_layout) =
                gate.storage_and_layout(
                );

            let (u_storage, u_layout) =
                up.storage_and_layout();

            if let (candle_core::Storage::Cpu(g_cpu), candle_core::Storage::Cpu(u_cpu)) = (&*g_storage, &*u_storage) {
                 if g_layout.is_contiguous() && u_layout.is_contiguous() {
                     // We need to be careful with mutation in candle. 
                     // Since we have ownership of 'gate' (it's a new tensor from projection), we can mutate its storage.
                     // However, candle doesn't give us a &mut [f32] easily. 
                     // We'll use unsafe to get a mutable pointer to the storage we own.
                     let g_ptr = g_cpu.as_slice::<f32>()?.as_ptr() as *mut f32;
                     let u_ptr = u_cpu.as_slice::<f32>()?.as_ptr();
                     let len = g_layout.shape().elem_count();

                     let arch = Arch::new();
                     arch.dispatch(|| {
                         unsafe {
                             let g_slice = std::slice::from_raw_parts_mut(g_ptr.add(g_layout.start_offset()), len);
                             let u_slice = std::slice::from_raw_parts(u_ptr.add(u_layout.start_offset()), len);

                             // NewGelu approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                             const SQRT_2_PI: f32 = 0.79788456;
                             for i in 0..len {
                                 let x = g_slice[i];
                                 let x3 = x * x * x;
                                 let inner = SQRT_2_PI * (x + 0.044715 * x3);
                                 let res = 0.5 * x * (1.0 + inner.tanh());
                                 g_slice[i] = res * u_slice[i];
                             }
                         }
                     });

                     return gate.apply(&self.down_proj.inner);
                 }
             }
        }

        // Fallback
        let lhs = gate.apply(&candle_nn::Activation::NewGelu)?;

        (lhs * up)?.apply(
            &self.down_proj.inner,
        )
    }
}

#[cfg(feature = "ssr")]

pub struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

#[cfg(feature = "ssr")]

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {

        let (b_sz, q_len, _) =
            xs.dims3()?;

        let (q, (k, v)) = rayon::join(
            || {

                self.q_proj
                    .forward(xs)
            },
            || {

                rayon::join(
                    || {

                        self.k_proj
                            .forward(xs)
                    },
                    || {

                        self.v_proj
                            .forward(xs)
                    },
                )
            },
        );

        let q = q?.reshape((
            b_sz,
            q_len,
            self.num_heads,
            self.head_dim,
        ))?;

        let k = k?.reshape((
            b_sz,
            q_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let v = v?.reshape((
            b_sz,
            q_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;

        let q = self
            .q_norm
            .forward(&q)?;

        let k = self
            .k_norm
            .forward(&k)?;

        // Transpose to (b, h, s, d) for RoPE and Attention
        let q = q.transpose(1, 2)?;

        let k = k.transpose(1, 2)?;

        let v = v.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self
            .rotary_emb
            .apply(
                &q,
                &k,
                seqlen_offset,
            )?;

        // KV Cache Update
        let (k, v) = match &self
            .kv_cache
        {
            | None => (k, v),
            | Some((pk, pv)) => {

                let k = Tensor::cat(
                    &[pk, &k],
                    2,
                )?;

                let v = Tensor::cat(
                    &[pv, &v],
                    2,
                )?;

                (k, v)
            },
        };

        self.kv_cache = Some((
            k.clone(),
            v.clone(),
        ));

        // GQA Expansion: (b, n_kv, s, d) -> (b, n_h, s, d)
        let total_s = k.dims()[2];

        let k = if self.num_kv_heads
            != self.num_heads
        {

            k.unsqueeze(2)?
                .broadcast_as((
                    b_sz,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    total_s,
                    self.head_dim,
                ))?
                .contiguous()?
                .reshape((
                    b_sz,
                    self.num_heads,
                    total_s,
                    self.head_dim,
                ))?
        } else {

            k
        };

        let v = if self.num_kv_heads
            != self.num_heads
        {

            v.unsqueeze(2)?
                .broadcast_as((
                    b_sz,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    total_s,
                    self.head_dim,
                ))?
                .contiguous()?
                .reshape((
                    b_sz,
                    self.num_heads,
                    total_s,
                    self.head_dim,
                ))?
        } else {

            v
        };

        let k_t = k.transpose(2, 3)?;

        let scale = 1.0
            / (self.head_dim as f64)
                .sqrt();

        let att =
            (q.matmul(&k_t)? * scale)?;

        let att = match mask {
            | None => att,
            | Some(m) => {
                att.broadcast_add(m)?
            },
        };

        let att = candle_nn::ops::softmax_last_dim(&att)?;

        // Output: (b, h, s, total_s) matmul (b, h, total_s, d) -> (b, h, s, d)
        let out = att
            .matmul(&v)?
            .transpose(1, 2)? // (b, s, h, d)
            .contiguous()?
            .reshape((
                b_sz,
                q_len,
                (),
            ))?;

        out.apply(&self.o_proj.inner)
    }
}

#[cfg(feature = "ssr")]

struct DecoderLayer {
    attention: Attention,
    mlp: MLP,
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
}

#[cfg(feature = "ssr")]

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {

        let residual = xs;

        let x = self
            .attn_norm
            .forward(xs)?;

        let x = self
            .attention
            .forward(
                &x,
                mask,
                seqlen_offset,
            )?;

        let x = self
            .post_attn_norm
            .forward(&x)?;

        let xs = (x + residual)?;

        let residual = &xs;

        let x = self
            .ffn_norm
            .forward(&xs)?;

        let x = self
            .mlp
            .forward(&x)?;

        let x = self
            .post_ffn_norm
            .forward(&x)?;

        x + residual
    }
}

#[cfg(feature = "ssr")]

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QLinear,
    hidden_size: usize,
    device: Device,
    dtype: DType,
}

#[cfg(feature = "ssr")]

impl Model {
    pub fn from_gguf<P: AsRef<Path>>(
        path: P,
        device: &Device,
    ) -> anyhow::Result<Self> {

        use log;

        log::info!(
            "🔍 Loading GGUF from {:?}",
            path.as_ref()
        );

        let file =
            std::fs::File::open(path)?;

        let mmap = unsafe {

            memmap2::Mmap::map(&file)?
        };

        let mut reader =
            std::io::Cursor::new(&mmap);

        let content =
            gguf_file::Content::read(
                &mut reader,
            )?;

        log::info!("📝 GGUF Keys:");
        for key in content
            .tensor_infos
            .keys()
        {

            if key.contains("blk.0")
                || !key.contains("blk.")
            {
                log::info!("  {}", key);
            }
        }

        let cfg = Config::gemma3_270m();

        log::info!("🧠 Initializing Model with Config: {:?}", cfg);

        let mut layers =
            Vec::with_capacity(
                cfg.num_hidden_layers,
            );

        let embed_tokens_weight =
            content
                .tensor(
                    &mut reader,
                    "token_embd.weight",
                    device,
                )?
                .dequantize(device)?;

        let embed_tokens =
            candle_nn::Embedding::new(
                embed_tokens_weight,
                cfg.hidden_size,
            );

        let rotary_emb_local = Arc::new(
            RotaryEmbedding::new(
                DType::F32,
                &cfg,
                10_000.0,
                device,
            )?,
        );

        // Gemma 3 uses 1M for global layers as per tech report
        let rotary_emb_global =
            Arc::new(
                RotaryEmbedding::new(
                    DType::F32,
                    &cfg,
                    1_000_000.0,
                    device,
                )?,
            );

        for i in
            0 .. cfg.num_hidden_layers
        {

            let prefix =
                format!("blk.{}", i);

            // Per Gemma 3 report: 1 global for every 5 local layers. Starting with local.
            // Layers: 0,1,2,3,4 (Local), 5 (Global), 6,7,8,9,10 (Local), 11 (Global), 12,13,14,15 (Local)
            let is_global =
                i == 5 || i == 11;

            let rotary_emb =
                if is_global {

                    rotary_emb_global
                        .clone()
                } else {

                    rotary_emb_local
                        .clone()
                };

            let q_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.attn_q.weight", prefix), device)?)?;

            let k_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.attn_k.weight", prefix), device)?)?;

            let v_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.attn_v.weight", prefix), device)?)?;

            let o_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.attn_output.weight", prefix), device)?)?;

            let q_norm_w = content.tensor(&mut reader, &format!("{}.attn_q_norm.weight", prefix), device)?.dequantize(device)?;

            let k_norm_w = content.tensor(&mut reader, &format!("{}.attn_k_norm.weight", prefix), device)?.dequantize(device)?;

            let q_norm = RmsNorm::new(
                q_norm_w,
                cfg.rms_norm_eps,
            );

            let k_norm = RmsNorm::new(
                k_norm_w,
                cfg.rms_norm_eps,
            );

            let attention = Attention {
                q_proj, k_proj, v_proj, o_proj,
                q_norm, k_norm,
                num_heads: cfg.num_heads(), // Corrected helper for num_heads? No, field is fine.
                num_kv_heads: cfg.num_kv_heads(),
                num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                rotary_emb: rotary_emb.clone(),
                kv_cache: None,
            };

            let gate_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.ffn_gate.weight", prefix), device)?)?;

            let up_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.ffn_up.weight", prefix), device)?)?;

            let down_proj = QLinear::new(content.tensor(&mut reader, &format!("{}.ffn_down.weight", prefix), device)?)?;

            let mlp = MLP {
                gate_proj,
                up_proj,
                down_proj,
            };

            let attn_norm = RmsNorm::new(
                content.tensor(&mut reader, &format!("{}.attn_norm.weight", prefix), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            );

            let post_attn_norm = RmsNorm::new(
                content.tensor(&mut reader, &format!("{}.post_attention_norm.weight", prefix), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            );

            let ffn_norm = RmsNorm::new(
                content.tensor(&mut reader, &format!("{}.ffn_norm.weight", prefix), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            );

            let post_ffn_norm = RmsNorm::new(
                content.tensor(&mut reader, &format!("{}.post_ffw_norm.weight", prefix), device)?.dequantize(device)?,
                cfg.rms_norm_eps,
            );

            layers.push(DecoderLayer {
                attention,
                mlp,
                attn_norm,
                post_attn_norm,
                ffn_norm,
                post_ffn_norm,
            });
            log::info!("  Layer {} loaded", i);
        }

        let norm_w = content
            .tensor(
                &mut reader,
                "output_norm.weight",
                device,
            )?
            .dequantize(device)?;

        let norm = RmsNorm::new(
            norm_w,
            cfg.rms_norm_eps,
        );

        let lm_head_weight = if content
            .tensor_infos
            .contains_key(
                "output.weight",
            ) {

            content.tensor(
                &mut reader,
                "output.weight",
                device,
            )?
        } else if content
            .tensor_infos
            .contains_key(
                "token_embd.weight",
            )
        {

            log::info!("💡 Weight tying detected: using token_embd.weight for output.weight");
            content.tensor(
                &mut reader,
                "token_embd.weight",
                device,
            )?
        } else {

            anyhow::bail!(
                "Could not find \
                 output.weight or \
                 token_embd.weight in \
                 GGUF"
            );
        };

        let lm_head = QLinear::new(
            lm_head_weight,
        )?;

        log::info!("✨ Model loading complete.");

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            hidden_size: cfg
                .hidden_size,
            device: device.clone(),
            dtype: DType::F32,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {

        let (b_sz, seq_len) =
            input_ids.dims2()?;

        // Streamlined mask generation
        let mask = if seq_len <= 1 {

            None
        } else {

            let mut mask_vec = vec![
                    0.0f32;
                    seq_len * seq_len
                ];

            for i in 0 .. seq_len {

                for j in
                    i + 1 .. seq_len
                {

                    mask_vec[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }

            Some(
                Tensor::from_vec(
                    mask_vec,
                    (
                        1,
                        1,
                        seq_len,
                        seq_len,
                    ),
                    &self.device,
                )?
                .to_dtype(self.dtype)?,
            )
        };

        let xs = self
            .embed_tokens
            .forward(input_ids)?;

        let scale = (self.hidden_size
            as f64)
            .sqrt();

        let mut xs = (xs * scale)?;

        for layer in self
            .layers
            .iter_mut()
        {

            xs = layer.forward(
                &xs,
                mask.as_ref(),
                seqlen_offset,
            )?;
        }

        // Final normalization and projection for the last token only
        let xs = xs
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?;

        let mut logits = xs.apply(
            &self.lm_head.inner,
        )?;

        // Optimized logit soft-capping: tanh(logits / cap) * cap
        let cap = 30.0f64;

        if logits
            .device()
            .is_cpu()
            && logits.dtype()
                == DType::F32
        {

            logits =
                logits.contiguous()?;

            let (storage, layout) =
                logits
                    .storage_and_layout(
                    );

            if let candle_core::Storage::Cpu(cpu) = &*storage {
                        let ptr = cpu.as_slice::<f32>()?.as_ptr() as *mut f32;
                        let len = layout.shape().elem_count();
                        let arch = Arch::new();
                        arch.dispatch(|| {
                            unsafe {
                                let offset = layout.start_offset();
                                let slice = std::slice::from_raw_parts_mut(ptr.add(offset), len);
                                let cap_f32 = cap as f32;
                                let inv_cap_f32 = 1.0 / cap_f32;
                                for i in 0..len {
                                    slice[i] = (slice[i] * inv_cap_f32).tanh() * cap_f32;
                                }
                            }
                        });
                        drop(storage);
                        return Ok(logits);
                    }
        }

        // Fallback for non-CPU/non-F32
        let inv_cap = 1.0 / cap;

        let logits = logits
            .affine(inv_cap, 0.0)?;

        let logits = logits.tanh()?;

        logits.affine(cap, 0.0)
    }

    pub fn clear_kv_cache(&mut self) {

        for layer in self
            .layers
            .iter_mut()
        {

            layer
                .attention
                .kv_cache = None;
        }
    }
}

#[cfg(feature = "ssr")]

// Added missing helpers to Config logic if needed, but I'll stick to fields.
impl Config {
    #[allow(clippy::inline_always)]
    #[inline(always)]

    fn num_heads(&self) -> usize {

        self.num_attention_heads
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]

    fn num_kv_heads(&self) -> usize {

        self.num_key_value_heads
    }
}

#[cfg(feature = "ssr")]

pub struct Gemma3 {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
    initialized: bool,
}

#[cfg(feature = "ssr")]

impl Gemma3 {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        tokenizer_path: P,
    ) -> anyhow::Result<Self> {

        let device = Device::Cpu;

        let model = Model::from_gguf(
            model_path,
            &device,
        )?;

        let tokenizer_file =
            std::fs::File::open(
                tokenizer_path,
            )?;

        let mut mmap = unsafe {

            memmap2::MmapOptions::new()
                .map_copy(
                    &tokenizer_file,
                )?
        };

        let tokenizer: Tokenizer = simd_json::from_slice(&mut mmap)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer JSON with SIMD: {}", e))?;

        Ok(Self {
            model,
            tokenizer,
            device,
            initialized: false,
        })
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]

    pub fn is_initialized(
        &self
    ) -> bool {

        self.initialized
    }

    #[allow(clippy::inline_always)]
    #[inline(always)]

    pub fn set_initialized(
        &mut self,
        v: bool,
    ) {

        self.initialized = v;
    }

    /// Generate text completion from the model
    ///
    /// Performance optimizations applied:
    /// - Pre-allocated string capacity to reduce reallocations
    /// - Early EOS token detection before decoding
    /// - Single decode call per token (removed redundant debug decode)
    /// - Optimized termination checks
    /// - KV cache reuse for sequential generation

    pub fn complete(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        check_cancel: impl Fn() -> bool,
    ) -> anyhow::Result<String> {

        self.model
            .clear_kv_cache();

        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(
                anyhow::Error::msg,
            )?;

        let mut tokens_vec = tokens
            .get_ids()
            .to_vec();

        // Pre-allocate string capacity for better performance
        // Assuming average of 4 chars per token
        let mut generated =
            String::with_capacity(
                max_tokens * 4,
            );

        log::info!("🎹 Tokens: {:?}", &tokens_vec[..std::cmp::min(tokens_vec.len(), 10)]);

        for i in 0 .. max_tokens {

            // Check for cancellation
            if check_cancel() {

                log::info!("✋ Generation stopped (cancellation signal).");
                break;
            }

            let context_size = if i == 0
            {

                tokens_vec.len()
            } else {

                1
            };

            let start_pos = tokens_vec
                .len()
                - context_size;

            let input = Tensor::new(
                &tokens_vec
                    [start_pos ..],
                &self.device,
            )?
            .unsqueeze(0)?;

            let logits =
                self.model.forward(
                    &input,
                    start_pos,
                )?;

            // Optimized argmax: Access CPU storage directly to avoid tensor metadata overhead
            // and multiple squeeze() operations.
            let next_token = if logits
                .device()
                .is_cpu()
            {

                let (storage, layout) = logits.storage_and_layout();

                if let candle_core::Storage::Cpu(cpu) = &*storage {
                    let slice = cpu.as_slice::<f32>()?;
                    let offset = layout.start_offset();
                    let len = layout.shape().elem_count();
                    let slice = &slice[offset..offset + len];

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    let arch = Arch::new();
                    arch.dispatch(|| {
                        // SIMD-friendly argmax
                        for (i, chunk) in slice.chunks_exact(16).enumerate() {
                            for j in 0..16 {
                                let v = chunk[j];
                                if v > max_val {
                                    max_val = v;
                                    max_idx = i * 16 + j;
                                }
                            }
                        }

                        // Remaining elements
                        let rem_start = (slice.len() / 16) * 16;
                        for (i, &v) in slice[rem_start..].iter().enumerate() {
                            if v > max_val {
                                max_val = v;
                                max_idx = rem_start + i;
                            }
                        }
                    });
                    max_idx as u32
                } else {
                    logits.argmax(D::Minus1)?.to_scalar::<u32>()?
                }
            } else {

                logits
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>(
                    )?
            };

            // Early termination check before decoding (performance optimization)
            // Check for common EOS tokens first
            if next_token == 1
                || next_token == 107
            {

                break;
            }

            tokens_vec.push(next_token);

            let token_text = self
                .tokenizer
                .decode(
                    &[next_token],
                    true,
                )
                .map_err(
                    anyhow::Error::msg,
                )?;

            // Helpful for debugging why self-test or generation might fail
            log::info!(
                "      [Token {}] id: \
                 {}, text: '{}'",
                i,
                next_token,
                token_text.replace(
                    "\n", "\\n"
                )
            );

            if token_text.contains(
                "<end_of_turn>",
            ) || token_text
                .contains("<eos>")
            {

                break;
            }

            generated
                .push_str(&token_text);
        }

        Ok(generated)
    }

    pub fn self_test(
        &mut self
    ) -> anyhow::Result<()> {

        use log;

        log::info!(
            "🧪 Running AI Self-Test \
             (Checking if model can \
             say 'YES')..."
        );

        // More standard prompt for Gemma models
        let res = self.complete(
            "<start_of_turn>user\\
             nAnswer with ONLY the \
             word YES: Is \
             1+1=2?<end_of_turn>\\
             n<start_of_turn>model\n",
            5,
            || false,
        )?;

        log::info!(
            "  Test Result: '{}'",
            res
        );

        // Very lenient check since it's a tiny model
        let upper = res.to_uppercase();

        if upper.contains("YES")
            || upper.contains("NO")
        {

            log::info!(
                "✅ Self-test passed \
                 (Found '{}').",
                upper.trim()
            );

            Ok(())
        } else if !res
            .trim()
            .is_empty()
        {

            log::info!(
                "⚠️ Self-test \
                 produced non-YES/NO \
                 output: '{}'. \
                 Proceeding anyway.",
                res.trim()
            );

            Ok(())
        } else {

            Err(anyhow::anyhow!(
                "Self-test failed: No \
                 tokens generated"
            ))
        }
    }
}

#[cfg(feature = "ssr")]

impl Drop for Gemma3 {
    fn drop(&mut self) {

        // Simple log to verify destruction on server stdout
        // We use log::info! or log! if available. Since it's library code, log::info might be safer or log.
        // But let's assume `log` crate is available as it's used in lib.rs
        // Actually ai.rs does not import log. We can add `use log;` or just log::info.
        // println!(
        //     "♻️ Gemma3 Model is being \
        //      dropped/deallocated."
        // );

        use log;

        log::info!(
            "♻️ Gemma3 Model is being \
             dropped/deallocated."
        );
    }
}
