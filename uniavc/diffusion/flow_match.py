import torch
import math
from typing_extensions import Literal


class FlowMatchScheduler():

    def __init__(self, template: Literal["FLUX.1", "Wan", "Qwen-Image", "FLUX.2", "Z-Image", "UniAVC", "Qwen-Image-Lightning", "ERNIE-Image", "ACE-Step"] = "FLUX.1"):
        self.set_timesteps_fn = {
            "FLUX.1": FlowMatchScheduler.set_timesteps_flux,
            "Wan": FlowMatchScheduler.set_timesteps_wan,
            "Qwen-Image": FlowMatchScheduler.set_timesteps_qwen_image,
            "FLUX.2": FlowMatchScheduler.set_timesteps_flux2,
            "Z-Image": FlowMatchScheduler.set_timesteps_z_image,
            "UniAVC": FlowMatchScheduler.set_timesteps_uniavc,
            "Qwen-Image-Lightning": FlowMatchScheduler.set_timesteps_qwen_image_lightning,
            "ERNIE-Image": FlowMatchScheduler.set_timesteps_ernie_image,
            "ACE-Step": FlowMatchScheduler.set_timesteps_ace_step,
            "HiDream-O1-Image": FlowMatchScheduler.set_timesteps_hidream_o1_image,
        }.get(template, FlowMatchScheduler.set_timesteps_flux)
        self.num_train_timesteps = 1000

    @staticmethod
    def set_timesteps_flux(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.003/1.002
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_wan(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def _calculate_shift_qwen_image(image_seq_len, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
    
    @staticmethod
    def set_timesteps_qwen_image(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Shift terminal
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def set_timesteps_qwen_image_lightning(num_inference_steps=100, denoising_strength=1.0, exponential_shift_mu=None, dynamic_shift_len=None):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        base_shift = math.log(3)
        max_shift = math.log(3)
        # Sigmas
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        # Mu
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._calculate_shift_qwen_image(dynamic_shift_len, base_shift=base_shift, max_shift=max_shift)
        else:
            mu = 0.8
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        # Timesteps
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps
    
    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666

        if image_seq_len > 4300:
            mu = a2 * image_seq_len + b2
            return float(mu)

        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1

        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b

        return float(mu)
    
    @staticmethod
    def set_timesteps_flux2(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        if dynamic_shift_len is None:
            # If you ask me why I set mu=0.8, I can only say that it yields better training results.
            mu = 0.8
        else:
            mu = FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ernie_image(num_inference_steps=50, denoising_strength=1.0, shift=3.0):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_ace_step(num_inference_steps=8, denoising_strength=1.0, shift=3.0):
        num_train_timesteps = 1000
        sigma_start = denoising_strength
        sigmas = torch.linspace(sigma_start, 0.0, num_inference_steps + 1)[:-1]
        if shift is not None and shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_z_image(num_inference_steps=100, denoising_strength=1.0, shift=None, target_timesteps=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 3 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        if target_timesteps is not None:
            target_timesteps = target_timesteps.to(dtype=timesteps.dtype, device=timesteps.device)
            for timestep in target_timesteps:
                timestep_id = torch.argmin((timesteps - timestep).abs())
                timesteps[timestep_id] = timestep
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_joyai_image(num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 4.0 if shift is None else shift
        num_train_timesteps = 1000
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def set_timesteps_hidream_o1_image(num_inference_steps=28, denoising_strength=1.0, shift=None, special_case=None, **kwargs):
        num_train_timesteps = 1000
        shift = 3.0 if shift is None else shift
        if special_case == "dev":
            timesteps_list = [
                999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
                707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
            ]
            sigmas = torch.tensor([t / 1000.0 for t in timesteps_list], dtype=torch.float32)
            timesteps = torch.tensor(timesteps_list, dtype=torch.float32)
            return sigmas, timesteps
        else:
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
            timesteps = sigmas * num_train_timesteps
            return sigmas, timesteps

    @staticmethod
    def _reference_curve(sigma_shift, shift, sigma_start, sigma_min, terminal, num_train_timesteps):
        """The n=1000 base sigma curve for one branch -- i.e. exactly the training grid.

        Shifted sampling grids are linear subsamples of this, so they cannot leave the
        curve the network was trained on.

        `shift` REPLACES the built-in warp rather than composing with it (Option A).
        That built-in warp IS a multiplicative shift: exp(2.05) = 7.7679 makes
        exp(m)/(exp(m) + (1/s - 1)) equal to k*s/(1 + (k-1)*s) at k = 7.7679 to 1.8e-07.
        So applying `shift` on top of it would not give shift 12/3 -- it would give
        7.7679*12 = 93.2 and 7.7679*3 = 23.3. At an effective shift of 93 the schedule
        sits above sigma 0.8 for ~90% of its length and spends almost no steps in the
        low-sigma range where detail is resolved, which shows up as blur. `sigma_shift`
        is therefore accepted and deliberately unused on this path, kept in the signature
        so the stock branch and this one read from one place.

        float32 throughout, matching the stock expression term for term, so the sampling
        grids stay on the same curve the n=1000 grid defines. float32 leaves them 4.6e-08
        off-curve, six orders below the 3.4e-02 that stock renorm produces and far below
        the 5.5e-03 spacing of the training grid itself.
        """
        ref = torch.linspace(sigma_start, sigma_min, num_train_timesteps + 1)[:-1]
        ref = shift * ref / (1 + (shift - 1) * ref)
        one_minus_z = 1.0 - ref
        scale_factor = one_minus_z[-1] / (1 - terminal)
        return 1.0 - (one_minus_z / scale_factor)

    @staticmethod
    def set_timesteps_uniavc(num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None, terminal=0.1, special_case=None, shift=None):
        """base sigma schedule, optionally with a MiniMax-H3 style per-branch `shift`.

        `shift=None` is the stock path and is bit-identical to what it has always been.

        `shift` is set only by the dual-scheduler (joint AV) setup, where video walks a
        shift=12.0 grid and audio a shift=3.0 grid, one timestep_id drawn per sample and BOTH
        grids indexed at it. The (sigma_video, sigma_audio) PAIR is a network input -- the AV
        cross-attention AdaLN is cross-wired, each branch's modulation built from the other
        branch's sigma -- so the pair seen at sampling must be the pair seen at training.

        Training runs at n=1000, so the network only ever saw the 1000 pairs
        {(sigma_v[i], sigma_a[i])}: a single 1-D CURVE in the (sigma_v, sigma_a) plane. Sampling
        at n=30 must therefore visit points ON that curve, or the AdaLN is modulated on a pair
        it was never trained for.

        `shift` REPLACES the built-in exp warp rather than composing with it, because that
        warp is itself a multiplicative shift at k = exp(2.05) = 7.7679. Composing would turn
        shift 12 / 3 into an effective 93.2 / 23.3, holding sigma above 0.8 for ~90% of the
        schedule and starving the low-sigma steps that resolve detail. See
        `_reference_curve`.

        Stock renorm cannot do that. `scale_factor = one_minus_z[-1] / (1 - terminal)` reads the
        LAST element of the curve being BUILT, and since sigmas is linspace(1, 0, n+1)[:-1] that
        element sits at fractional position n/(n+1) -- its place on the curve moves with n. The
        factor varies with step count and differs per branch. It pins both endpoints and
        stretches everything between, so the whole middle of the schedule lifts off the trained
        curve and the worst sigma_audio error against the training pair drifts 0.053 at n=30 /
        0.034 at n=50. n=30 and n=50 denoise along genuinely different curves.

        This builds the FIXED n=1000 reference curve -- terminal renorm included, so it is
        exactly the training grid -- and then SUBSAMPLES it by linear interpolation onto
        `num_inference_steps` points. The curve is n-invariant by construction (worst pair
        error 1.1e-03 at n=30, 51x better than stock renorm, and flat in n rather than growing
        as n shrinks) and every grid still lands on the terminal exactly.

        The rejected alternative was to keep the stock formula but read `one_minus_z[-1]` off the
        n=1000 reference (n-invariant renorm). That is also on-curve, but it does not reach the
        terminal at low n -- sigma_v ends at 0.3558 for n=30 and 0.2684 for n=50 instead of 0.1,
        leaving a third of the noise unremoved. Subsampling is the only scheme that is on-curve,
        n-invariant, AND terminal-correct at the same time.

        At n=1000 the interpolation is the identity, so the training grid is bit-identical to
        both the old code and the stock schedule.
        """
        num_train_timesteps = 1000
        if special_case == "stage2":
            sigmas = torch.Tensor([0.909375, 0.725, 0.421875])
        elif special_case == "ditilled_stage1":
            sigmas = torch.Tensor([1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875])
        else:
            dynamic_shift_len = dynamic_shift_len or 4096
            sigma_shift = FlowMatchScheduler._calculate_shift_qwen_image(
                image_seq_len=dynamic_shift_len,
                base_seq_len=1024,
                max_seq_len=4096,
                base_shift=0.95,
                max_shift=2.05,
            )
            sigma_min = 0.0
            sigma_max = 1.0
            sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
            if shift is None:
                sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
                sigmas = math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1))
                # Shift terminal
                one_minus_z = 1.0 - sigmas
                scale_factor = one_minus_z[-1] / (1 - terminal)
                sigmas = 1.0 - (one_minus_z / scale_factor)
            else:
                # Per-branch shift, subsampled off the n=1000 training curve so the
                # (sigma_video, sigma_audio) pair stays on the trained manifold at any step count.
                sigmas = FlowMatchScheduler._reference_curve(
                    sigma_shift=sigma_shift,
                    shift=shift,
                    sigma_start=sigma_start,
                    sigma_min=sigma_min,
                    terminal=terminal,
                    num_train_timesteps=num_train_timesteps,
                )
                if num_inference_steps != num_train_timesteps:
                    # float64 only for the index weights: 1/999 is not representable in float32.
                    idx = torch.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=torch.float64)
                    lo = idx.floor().long()
                    hi = idx.ceil().long()
                    w = (idx - lo.to(idx.dtype)).to(sigmas.dtype)
                    sigmas = sigmas[lo] * (1.0 - w) + sigmas[hi] * w
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    def set_training_weight(self):
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    
    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    def add_noise(self, original_samples, noise, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    
    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    
    def training_weight(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class HiDreamO1FlashScheduler(FlowMatchScheduler):
    
    def __init__(self, noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5):
        self.set_timesteps_fn = HiDreamO1FlashScheduler.set_timesteps_hidream_o1_image_dev
        self.num_train_timesteps = 1000
        self.noise_clip_std = noise_clip_std
        num_steps = 28
        self.noise_scale_schedule = [
            noise_scale_start + (noise_scale_end - noise_scale_start) * i / (num_steps - 1)
            for i in range(num_steps)
        ]

    @staticmethod
    def set_timesteps_hidream_o1_image_dev(**kwargs):
        timesteps_list = [
            999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737,
            707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8,
        ]
        sigmas = torch.tensor([t / 1000.0 for t in timesteps_list], dtype=torch.float32)
        timesteps = torch.tensor(timesteps_list, dtype=torch.float32)
        return sigmas, timesteps

    def clip_noise(self, noise):
        if self.noise_clip_std > 0:
            noise_std = noise.std().item()
            clip_val = self.noise_clip_std * noise_std
            noise = noise.clamp(min=-clip_val, max=clip_val)
        return noise
    
    def step(self, model_output, timestep, sample):
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_ = self.sigmas[timestep_id + 1] if timestep_id + 1 < len(self.sigmas) else 0
        denoised = sample - model_output * sigma

        noise = self.clip_noise(torch.randn(denoised.shape, device=denoised.device, dtype=denoised.dtype))
        sample = sigma_ * noise * self.noise_scale_schedule[timestep_id] + (1.0 - sigma_) * denoised
        return sample
