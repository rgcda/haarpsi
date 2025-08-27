#=
haar_psi.jl
Julia language version of HaarPSI

Translated from Matlab code written by Rafael Reisenhofer circa 08/05/2017
=#

using DSP: conv

# export haar_psi, haar_psi_maps


"""
    (similarity, similarityMaps, weightMaps) =
         haar_psi_maps(imgRef, imgDis, preprocessWithSubsampling=1)

Compute HaarPSI,
the Haar wavelet-based perceptual similarity index of two images.

Make sure that grayscale and color values are given in the [0,255] interval!
If this is not the case, the HaarPSI cannot be computed correctly.

Input:

- `imgRef`: reference RGB or grayscale image with values ranging from 0 to 255.
- `imgDis`: distorted RGB or grayscale image with values ranging from 0 to 255.

Option:
- `preprocessWithSubsampling`: If `false`, the preprocssing step to accommodate
  for the viewing distance in psychophysical experiments is omitted.


Output:
- `similarity`: The Haar wavelet-based perceptual similarity index,
  measured in the interval [0,1].
- `similarityMaps`: Maps of horizontal and vertical local similarities.
  For RGB images, this variable also includes a similarity map
  with respect to the two color channesl in the YIQ space.
- `weightMaps`: Weight maps measuring the importance
  of the local similarities in similarityMaps.

Example:

imgRef = load("peppers.png")
imgDis = imgRef + rand(-20:20, size(imgRef))
imgDis = clamp.(imgDis, 0, 255)
similarity, sim_map, wt_map = haar_psi_maps(imgRef, imgDis)

Reference:

R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand:
'A Haar Wavelet-Based Perceptual Similarity Index
for Image Quality Assessment', 2017.
https://doi.org/10.1016/j.image.2017.11.001

todo: handle RGB type properly
"""
haar_psi_maps(
    imgRef::AbstractMatrix{<:Number},
    imgDis::AbstractMatrix{<:Number},
    preprocessWithSubsampling::Bool = true,
) = HaarPSI(Float64.(imgRef), Float64.(imgDis), preprocessWithSubsampling)

function haar_psi_maps(
    imgRef::AbstractMatrix{<:AbstractFloat},
    imgDis::AbstractMatrix{<:AbstractFloat},
    preprocessWithSubsampling::Bool = true;
    C::Real = 30,
    alpha::Real = 4.2,
)

    colorImage = size(imgRef, 3) == 3

    # initialization and preprocessing

    # transform to YIQ colorspace
    if colorImage
        imgRefY = 0.299 * (imgRef[:,:,1]) + 0.587 * (imgRef[:,:,2]) + 0.114 * (imgRef[:,:,3])
        imgDisY = 0.299 * (imgDis[:,:,1]) + 0.587 * (imgDis[:,:,2]) + 0.114 * (imgDis[:,:,3])
        imgRefI = 0.596 * (imgRef[:,:,1]) - 0.274 * (imgRef[:,:,2]) - 0.322 * (imgRef[:,:,3])
        imgDisI = 0.596 * (imgDis[:,:,1]) - 0.274 * (imgDis[:,:,2]) - 0.322 * (imgDis[:,:,3])
        imgRefQ = 0.211 * (imgRef[:,:,1]) - 0.523 * (imgRef[:,:,2]) + 0.312 * (imgRef[:,:,3])
        imgDisQ = 0.211 * (imgDis[:,:,1]) - 0.523 * (imgDis[:,:,2]) + 0.312 * (imgDis[:,:,3])
    else
        imgRefY = imgRef
        imgDisY = imgDis
    end

    # subsampling
    if preprocessWithSubsampling
        imgRefY = haar_psi_subsample(imgRefY)
        imgDisY = haar_psi_subsample(imgDisY)
        if colorImage
            imgRefQ = haar_psi_subsample(imgRefQ)
            imgDisQ = haar_psi_subsample(imgDisQ)
            imgRefI = haar_psi_subsample(imgRefI)
            imgDisI = haar_psi_subsample(imgDisI)
        end
    end

    # pre-allocate variables
    if colorImage
        localSimilarities = zeros(size(imgRefY)..., 3)
        weights = zeros(size(imgRefY)..., 3)
    else
        localSimilarities = zeros(size(imgRefY)..., 2)
        weights = zeros(size(imgRefY)..., 2)
    end

    # Haar wavelet decomposition
    nScales = 3
    coeffsRefY = haar_psi_dec(imgRefY, nScales)
    coeffsDistY = haar_psi_dec(imgDisY, nScales)
    if colorImage
        coeffsRefQ = abs.(conv(imgRefQ, ones(2,2)/4))
        coeffsDistQ = abs.(conv(imgDisQ, ones(2,2)/4))
        coeffsRefI = abs.(conv(imgRefI, ones(2,2)/4))
        coeffsDistI = abs.(conv(imgDisI, ones(2,2)/4))
    end

    # compute weights and similarity for each orientation
    for ori in 1:2
        weights[:,:,ori] = max(
            abs.(coeffsRefY[:, :, 3 + (ori-1)*nScales]),
            abs.(coeffsDistY[:, :, 3 + (ori-1)*nScales]),
        )
        coeffsRefYMag = abs.(coeffsRefY[:,:,(1:2) + (ori-1)*nScales])
        coeffsDistYMag = abs.(coeffsDistY[:,:,(1:2) + (ori-1)*nScales])
        localSimilarities[:,:,ori] = sum((2*coeffsRefYMag.*coeffsDistYMag + C) ./
            (coeffsRefYMag.^2 + coeffsDistYMag.^2 + C), dims=3) / 2
    end

    # compute similarities for color channels
    if colorImage
        similarityI = (2*coeffsRefI.*coeffsDistI + C) ./ (coeffsRefI.^2 + coeffsDistI.^2 + C)
        similarityQ = (2*coeffsRefQ.*coeffsDistQ + C) ./ (coeffsRefQ.^2 + coeffsDistQ.^2 + C)
        localSimilarities[:,:,3] = (similarityI + similarityQ) / 2
        weights[:,:,3] = (weights[:,:,1] + weights[:,:,2]) / 2
    end

    # compute final score
    similarity = haar_psi_log_inv(
            sum(HaarPSILog(localSimilarities, alpha) .* weights) /
            sum(weights),
        alpha)^2

    # output maps
    return similarity, localSimilarities, weights
end

function haar_psi_dec(X::AbstractMatrix{<:AbstractFloat}, nScales::Int)
    coeffs = zeros(size(X)..., 2*nScales)
    for k = 1:nScales
        haarFilter = 2^(-k) * ones(2^k, 2^k)
        haarFilter[1:(end÷2),:] = -haarFilter[1:(end÷2),:]
        coeffs[:,:,k] = conv(X, haarFilter)
        coeffs[:,:,k + nScales] = conv(X, haarFilter')
    end
    return coeffs
end

function haar_psi_subsample(img::AbstractMatrix{<:AbstractFloat})
    imgSubsampled = conv(img, ones(2,2)/4)
    imgSubsampled = imgSubsampled[1:2:end,1:2:end]
    return imgSubsampled
end

haar_psi_log(x, alpha) = 1 / (1 + exp(-alpha * x))

haar_psi_log_inv(x, alpha) = log(x / (1-x)) / alpha
