import numpy as np
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from denoisers.kernel_filters import nlm as nlm

def proj(im_input, minval, maxval):
    im_out = np.where(im_input > maxval, maxval, im_input)
    im_out = np.where(im_out < minval, minval, im_out)
    return im_out

def psnr(x,im_orig):
    norm2 = np.mean((x - im_orig) ** 2)
    psnr = -10 * np.log10(norm2)
    return psnr

def funcAtranspose(im_input, mask, fx, fy):
    m,n = im_input.shape
    fx = int(1/fx)
    fy = int(1/fy)
    im_inputres = np.zeros([m*fx, m*fy], im_input.dtype)
    for i in range(m):
        for j in range(n):
            im_inputres[fx*i,fy*j] = im_input[i,j]
 
    m,n = im_inputres.shape
    w = len(mask[0])
    r = int((w - 1) / 2)
    im_inputres = cv2.copyMakeBorder(im_inputres, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_output = cv2.filter2D(im_inputres, -1, mask)
    im_output = im_output[r:r+m, r:r+n]
    return im_output

def funcA(im_input, mask, fx, fy):
    m,n = im_input.shape
    w = len(mask[0])
    r = int((w - 1) / 2)
    im_input = cv2.copyMakeBorder(im_input, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_output = cv2.filter2D(im_input, -1, mask)
    im_output = im_output[r:r+m, r:r+n]
    im_outputres = cv2.resize(im_output, (0,0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    return im_outputres

def pnp_fbs_superresolution(im_input, im_ref, fx, fy, mask, **opts):

    lamda = opts.get('lamda', 2.0)
    rho = opts.get('rho', 1.0)
    maxitr = opts.get('maxitr', 100)
    verbose = opts.get('verbose',1)
    sigma = opts.get('sigma', 5)

    """ Initialization. """
    index = np.nonzero(mask)
    y = funcAtranspose(im_input, mask, fx, fy)
    m, n = y.shape

    x = cv2.resize(im_input, (m, n))
    ytilde = x
    res = np.zeros(maxiter)
    funceval = np.zeros(maxiter)
    """ Main loop. """
    for i in range(maxitr):

        xold = np.copy(x)

        """ Update gradient. """
        xoldhat = funcA(x, mask, fx, fy)
        gradx = funcAtranspose(xoldhat, mask, fx, fy) - y

        """ Denoising step. """
        if i<=5:        # Warm-up for first 5 iterations    
            xtilde = np.copy(xold - 8. * gradx)
            ytilde = xtilde
            out = nlm(xtilde, ytilde, patch_rad=3, window_rad=3, sigma=10*sigma/255.)
            x = out[0]
            D = out[1]
            dmin = np.amin(D)
            D = (1. / dmin) * D
            D = 1./D
        else:
            xtilde = np.copy(xold - rho * np.multiply(D,gradx))
            out = nlm(xtilde, ytilde, patch_rad=3, window_rad=3, sigma=2*sigma/255.)
            x = out[0]
        
        res[i] = np.linalg.norm(x - xold)
        funceval[i] = 0.5 * np.linalg.norm(im_input - funcA(x, mask, fx, fy))
        funceval[i] = funceval[i] + (1./rho)*np.sum(np.multiply(np.multiply(D,x), xtilde -x))
        """ Monitoring. """
        if verbose:
            print("i: {}, \t psnr: {} ssim= {} "\
                  .format(i+1, psnr(x,im_ref), compare_ssim(x, im_ref, data_range=1.)))

    return x

if __name__ == "__main__":
    dir_name = 'testsets/Set12/'
    outdir_name = 'output_images/'
    imagename = '11.png'
    input_str = dir_name + imagename
    K = 2 # downsampling factor
    # ---- load the ground truth ----
    im_orig = cv2.imread(input_str, 0)/255.0
    m,n = im_orig.shape

    # ---- blur the image 
    kernel = cv2.getGaussianKernel(9, 1)
    mask = np.outer(kernel, kernel.transpose())
    w = len(mask[0])
    r = int((w - 1) / 2)
    im_orig = cv2.copyMakeBorder(im_orig, r, r, r, r, borderType=cv2.BORDER_WRAP)
    im_blur = cv2.filter2D(im_orig, -1, mask)
    im_blur = im_blur[r:r+m, r:r+n]
    im_orig = im_orig[r:r+m, r:r+m]

    # ---- Downsample the image
    fx = 1./K
    fy = 1./K 
    im_down = cv2.resize(im_blur, (0,0), fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)

    # ---- add noise -----
    noise_level = 10.0 / 255.0
    gauss = np.random.normal(0.0, noise_level, im_down.shape)
    im_noisy = im_down + gauss

    im_noisy2 = np.clip(im_noisy, 0., 1.)
    cv2.imwrite('observed.png', im_noisy2*255.)

    psnr_final = 0.

    # ---- set options -----
    sigma = 20
    rho = 0.05
    maxiter = 20

    bicubic_img = cv2.resize(im_noisy, None, fx = K, fy = K, interpolation = cv2.INTER_CUBIC)
    psnr_ours = psnr(bicubic_img, im_orig)
    ssim_ours = compare_ssim(bicubic_img, im_orig, data_range=1.)
    print('sigma = {}, rho = {} - PNSR: {}, SSIM = {}'.format(sigma, rho, psnr_ours, ssim_ours))

    opts = dict(sigma = sigma, rho = rho, maxitr = maxiter, verbose = True)

    # ---- plug and play -----
    out = pnp_fbs_superresolution(im_noisy, im_orig, fx, fy, mask, **opts)
    output_str = outdir_name + imagename
    cv2.imwrite(output_str, out * 255.0)
 
    # ---- results ----
     
    psnr_ours = psnr(out, im_orig)
    ssim_ours = compare_ssim(out, im_orig, data_range=1.)
    print('sigma = {}, rho = {} - PNSR: {}, SSIM = {}'.format(sigma, rho, psnr_ours, ssim_ours))


