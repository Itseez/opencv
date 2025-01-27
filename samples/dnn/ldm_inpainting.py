import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from download_models import downloadFile, getSaveDir
import yaml
import os

## let use write description of the script and general information how to use it

'''
This sample proposes experimental inpainting sample using Latent Diffusion Model (LDM) for inpainting.
Most of the script is based on the code from the official repository of the LDM model: https://github.com/CompVis/latent-diffusion

Current limitations of the script:
    - Slow diffusion sampling
    - Not exact reproduction of the results from the original repository (due to issues related deviation in covolution operation.
    See issue for more details: https://github.com/opencv/opencv/pull/25973)

Steps for running the script:

1. Firstly generate ONNX graph of the Latent Diffusion Model.

    Generate the using this [repo](https://github.com/Abdurrahheem/latent-diffusion/tree/ash/export2onnx) and follow instructions below

    - git clone https://github.com/Abdurrahheem/latent-diffusion.git
    - cd latent-diffusion
    - conda env create -f environment.yaml
    - conda activate ldm
    - wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
    - python -m scripts.inpaint.py --indir data/inpainting_examples/ --outdir outputs/inpainting_results --export=True

2. Build opencv (preferebly with CUDA support enabled
3. Run the script

    - cd opencv/samples/dnn
    - python ldm_inpainting -e=<path-to-InpaintEncoder.onnx file> -d=<path-to-InpaintDecoder.onnx file> -df=<path-to-LatenDiffusion.onnx file> -i=<path-to-image>

Right after the last command you will be promted with image. You can click on left mouse botton and starting selection a region you would like to be inpainted (delited).
Once you finish marking the region, click on left mouse botton again and press esc botton on your keyboard. The inpainting proccess will start.

Note: If you are running it on CPU it might take a large chank of time.
Also make sure to have abount 15GB of RAM to make proccess faster (other wise swapping will ckick in and everything will be slower)
'''


backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_CUDA)
targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL, cv.dnn.DNN_TARGET_CUDA)

parser = argparse.ArgumentParser(description='Use this script to run inpainting using Latent Diffusion Model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--encoder', '-e', type=str, help='Path to encoder network.', default=None)
parser.add_argument('--decoder', '-d', type=str, help='Path to decoder network.', default=None)
parser.add_argument('--diffusor', '-df', type=str, help='Path to diffusion network.', default=None)
parser.add_argument('--image', '-i', type=str, help='Path to input image.', default=None)
parser.add_argument('--samples', '-s', type=int, help='Number of times to sample the model.', default=50)
parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                             "%d: automatically (by default), "
                             "%d: OpenCV implementation, "
                             "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                             "%d: CUDA, " % backends)
parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                         '%d: CPU target (by default), '
                         '%d: OpenCL, '
                         '%d: NCS2 VPU, '
                         '%d: HDDL VPU, '
                         '%d: CUDA ' % targets)
parser.add_argument('--mask', '-m', type=str, help='Path to mask image. If not provided, interactive mask creation will be used.', default=None)

def make_batch(image, mask):
    image = image.astype(np.float32)/255.0
    image = image[np.newaxis, ...].transpose(0,3,1,2)

    mask = mask.astype(np.float32)/255.0
    mask = mask[np.newaxis, np.newaxis, ...]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    masked_image = (1-mask)*image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k]*2.0-1.0
    return batch

def noise_like(shape, repeat=False):
    repeat_noise = lambda: np.random.randn((1, *shape[1:])).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: np.random.randn(*shape)
    return repeat_noise() if repeat else noise()

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep).astype(np.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                np.arange(n_timestep + 1).astype(np.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start, linear_end, n_timestep).astype(np.float64)
    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end, n_timestep).astype(np.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", ddpm_num_timesteps=1000):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = ddpm_num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_numpy = partial(np.array, copy=True, dtype=np.float32)

        self.register_buffer('betas', to_numpy(self.model.betas))
        self.register_buffer('alphas_cumprod', to_numpy(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_numpy(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_numpy(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_numpy(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_numpy(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_numpy(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_numpy(np.sqrt(1. / alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * np.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               eta=0.,
               temperature=1.,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    ddim_use_original_steps=False,
                                                    temperature=temperature,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      timesteps=None,log_every_t=100, temperature=1.,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        b = shape[0]
        if x_T is None:
            img = np.random.randn(*shape)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = np.full((b, ), step, dtype=np.int64)

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      temperature=temperature, unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False,
                      temperature=1., unconditional_guidance_scale=1., unconditional_conditioning=None):
        b = x.shape[0]
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = np.full((b, 1, 1, 1), alphas[index])
        a_prev = np.full((b, 1, 1, 1), alphas_prev[index])
        sigma_t = np.full((b, 1, 1, 1), sigmas[index])
        sqrt_one_minus_at = np.full((b, 1, 1, 1), sqrt_one_minus_alphas[index])

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / np.sqrt(a_t)
        # direction pointing to x_t
        dir_xt = np.sqrt(1. - a_prev - sigma_t**2) * e_t
        noise = sigma_t * noise_like(x.shape, repeat_noise) * temperature
        x_prev = np.sqrt(a_prev) * pred_x0 + dir_xt + noise
        return x_prev, pred_x0


class DDIMInpainter(object):
    def __init__(self,
                 args,
                 v_posterior=0., # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 parameterization="eps",  # all assuming fixed variance schedules
                 linear_start=0.0015,
                 linear_end=0.0205,
                 conditioning_key="concat",
                 ):
        super().__init__()

        self.v_posterior = v_posterior
        self.parameterization = parameterization
        self.conditioning_key = conditioning_key
        self.register_schedule(linear_start=linear_start, linear_end=linear_end)

        # Load model configurations from YAML
        config_file = 'models.yml'
        save_dir = getSaveDir()
        models_info = yaml.safe_load(open(config_file, 'r'))
        ldm_config = models_info.get('ldm_inpainting', {})

        # Initialize models using provided paths or download if necessary
        encoder_path = args.encoder
        decoder_path = args.decoder
        diffusor_path = args.diffusor

        if not encoder_path and ldm_config.get('encoder_load_info'):
            encoder_path = os.path.join(save_dir, ldm_config['encoder_model'])
            if not os.path.exists(encoder_path):
                print(f"Could not find {ldm_config['encoder_model']} model graph at {args.encoder}. Downloading...")
                downloadFile(ldm_config['encoder_load_info']['url'],
                        sha=ldm_config['encoder_load_info']['sha1'],
                        save_dir=save_dir)

        if not decoder_path and ldm_config.get('decoder_load_info'):
            decoder_path = os.path.join(save_dir, ldm_config['decoder_model'])
            if not os.path.exists(decoder_path):
                print(f"Could not find {ldm_config['decoder_model']} model graph at {args.decoder}. Downloading...")
                downloadFile(ldm_config['decoder_load_info']['url'],
                           sha=ldm_config['decoder_load_info']['sha1'],
                           save_dir=save_dir)

        if not diffusor_path and ldm_config.get('diffusor_load_info'):
            diffusor_path = os.path.join(save_dir, ldm_config['diffusor_model'])
            if not os.path.exists(diffusor_path):
                print(f"Could not find {ldm_config['diffusor_model']} model graph at {args.diffusor}. Downloading...")
                downloadFile(ldm_config['diffusor_load_info']['url'],
                           sha=ldm_config['diffusor_load_info']['sha1'],
                           save_dir=save_dir)

        # Initialize models
        self.encoder = cv.dnn.readNet(encoder_path)
        self.decoder = cv.dnn.readNet(decoder_path)
        self.diffusor = cv.dnn.readNet(diffusor_path)
        self.sampler = DDIMSampler(self, ddpm_num_timesteps=self.num_timesteps)
        self.set_backend(backend=args.backend, target=args.target)

    def set_backend(self, backend=cv.dnn.DNN_BACKEND_DEFAULT, target=cv.dnn.DNN_TARGET_CPU):
        self.encoder.setPreferableBackend(backend)
        self.encoder.setPreferableTarget(target)

        self.decoder.setPreferableBackend(backend)
        self.decoder.setPreferableTarget(target)

        self.diffusor.setPreferableBackend(backend)
        self.diffusor.setPreferableTarget(target)

    def apply_diffusor(self, x, timestep, cond):

        x = np.concatenate([x, cond], axis=1)
        x = cv.Mat(x.astype(np.float32))
        timestep = cv.Mat(timestep.astype(np.int64))
        names = ["xc, t", "timesteps"]
        self.diffusor.setInputsNames(names)
        self.diffusor.setInput(x, names[0])
        self.diffusor.setInput(timestep, names[1])
        output = self.diffusor.forward()
        return output

    def register_buffer(self, name, attr):
        setattr(self, name, attr)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_numpy = partial(np.array, dtype=np.float32)

        self.register_buffer('betas', to_numpy(betas))
        self.register_buffer('alphas_cumprod', to_numpy(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_numpy(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_numpy(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_numpy(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_numpy(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_numpy(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_numpy(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_numpy(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_numpy(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_numpy(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_numpy(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_numpy(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(alphas_cumprod) / (2. * 1 - alphas_cumprod)
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights)
        assert not np.isnan(self.lvlb_weights).all()

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            # if not isinstance(cond, list):
            #     cond = [cond]
            key = 'c_concat' if self.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.apply_diffusor(x_noisy, t, cond['c_concat'])

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def inpaint(self, image : np.ndarray, mask : np.ndarray, S : int = 50) -> np.ndarray:
        inpainted = self(image, mask, S)
        return np.squeeze(inpainted)

    def __call__(self, image : np.ndarray, mask : np.ndarray, S : int = 50) -> np.ndarray:

        # Encode the image and mask
        self.encoder.setInput(image)
        c = self.encoder.forward()
        cc = cv.resize(np.squeeze(mask), dsize=(c.shape[3], c.shape[2]), interpolation=cv.INTER_NEAREST) #TODO:check for correcteness of intepolation
        cc = cc[None,None]
        c = np.concatenate([c, cc], axis=1)

        shape = (c.shape[1] - 1,) + c.shape[2:]

        # Sample from the model
        samples_ddim, _ = self.sampler.sample(
            S=S,
            conditioning=c,
            batch_size=c.shape[0],
            shape=shape,
            verbose=False)

        ## Decode the sample
        samples_ddim = samples_ddim.astype(np.float32)
        samples_ddim = cv.Mat(samples_ddim)
        self.decoder.setInput(samples_ddim)
        x_samples_ddim = self.decoder.forward()

        image = np.clip((image + 1.0) / 2.0, a_min=0.0, a_max=1.0)
        mask = np.clip((mask + 1.0) / 2.0, a_min=0.0, a_max=1.0)
        predicted_image = np.clip((x_samples_ddim + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        inpainted = (1 - mask) * image + mask * predicted_image
        inpainted = np.transpose(inpainted, (0, 2, 3, 1)) * 255

        return inpainted

def create_mask(img, radius=20):
    drawing = False  # True if the mouse is pressed
    counter = 0

    # Mouse callback function
    def draw_circle(event, x, y, flags, param):
        nonlocal drawing, counter, radius

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True if counter % 2 == 0 else False
            counter += 1
            cv.circle(img, (x, y), radius, (255, 255, 255), -1)
            cv.circle(mask, (x, y), radius, 255, -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                cv.circle(img, (x, y), radius, (255, 255, 255), -1)
                cv.circle(mask, (x, y), radius, 255, -1)

    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    # Create window with instructions
    window_name = 'image - Controls: Left click to start/stop drawing, ESC to finish'
    cv.namedWindow(window_name)
    cv.setMouseCallback(window_name, draw_circle)

    while True:
        # Create a copy of the image to show instructions
        display_img = img.copy()
        if not drawing:
            cv.putText(display_img, 'Click to start drawing', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv.putText(display_img, 'Click to stop drawing', (10, 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.putText(display_img, 'Press ESC when finished', (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv.imshow(window_name, display_img)
        if cv.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cv.destroyAllWindows()
    return mask

def prepare_input(args):
    image = cv.imread(args.image)
    if args.mask:
        mask = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask file: {args.mask}")
        if mask.shape[:2] != image.shape[:2]:
            mask = cv.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
    else:
        mask = create_mask(deepcopy(image))

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    batch = make_batch(image, mask)
    return batch

def main(args):

    batch = prepare_input(args)

    model = DDIMInpainter(args)
    result = model.inpaint(batch["masked_image"], batch["mask"], S=args.samples)

    ext = args.image.split('.')[-1]
    cv.imwrite(args.image.replace(f".{ext}", f"_inpainted.{ext}"), result[..., ::-1])

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
