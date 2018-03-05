function [similarity,similarityMaps,weightMaps] = HaarPSIExt(imgRef,imgDist,preprocessWithSubsampling,wavelet,boundaryTreatment)
%HaarPSIExt Computes the Haar wavelet-based perceptual similarity index of two
%images.
%
%Please make sure that grayscale and color values are given in the [0,255]
%interval! If this is not the case, the HaarPSI cannot be computed
%correctly.
%
%Usage (optional parameters in <>-brackets):
%
% [similarity,<similarityMaps>,<weightMaps>] = HaarPSI(imgRef, imgDist, <preprocessWithSubsampling>, <wavelet>, <boundaryTreatment>);
%
%
%
%Input:
%
%                       imgRef: RGB or grayscale image with values ranging from 0
%                               to 255.
%                      imgDist: RGB or grayscale image with values ranging from 0
%                               to 255.
%    preprocessWithSubsampling: <optional> If 0, the preprocssing step to acommodate for the 
%                               viewing distance in psychophysical experimentes is omitted.
%                      wavelet: <optional> A string specifying the wavelet
%                               used in the definition of the perceptual
%                               simlarity measure. For a list of possible
%                               choices, see the documentation of wfilters.
%            boundaryTreatment: <optional> A string or number specifying
%                               boundary treatment. For possible choices,
%                               see the documentation of imfilter.
%                               
%                  
%
%Output:
%
%                   similarity: The Haar wavelet-based perceptual similarity index, measured
%                               in the interval [0,1].
%               similarityMaps: <optional> Maps of horizontal and vertical local similarities.
%                               For RGB images, this variable also includes
%                               a similarity map with respect to the two
%                               color channesl in the YIQ space.
%                   weightMaps: <optional> Weight maps measuring the importance of
%                               the local similarities in similarityMaps.
%                               
%
%Example:
%
% imgRef = double(imread('peppers.png'));
% imgDist = imgRef + randi([-20,20],size(imgRef));
% imgDist = min(max(imgDist,0),255);
% similarity = HaarPSI(imgRef,imgDist);
%
%Reference: 
%
% R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand: 'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment', 2017.
    if nargin < 3
        preprocessWithSubsampling = 1;
    end
    if nargin < 4
        wavelet = 'haar';
    end
    if nargin < 5
        boundaryTreatment = 0;
    end
    colorImage = size(imgRef,3) == 3;    
        
    imgRef = double(imgRef);
    imgDist = double(imgDist);
    
    %% initialization and preprocessing   
    %constants
    C = 30;
    alpha = 4.2;
    %transform to YIQ colorspace
    if colorImage        
        imgRefY = 0.299 * (imgRef(:,:,1)) + 0.587 * (imgRef(:,:,2)) + 0.114 * (imgRef(:,:,3));
        imgDistY = 0.299 * (imgDist(:,:,1)) + 0.587 * (imgDist(:,:,2)) + 0.114 * (imgDist(:,:,3));
        imgRefI = 0.596 * (imgRef(:,:,1)) - 0.274 * (imgRef(:,:,2)) - 0.322 * (imgRef(:,:,3));
        imgDistI = 0.596 * (imgDist(:,:,1)) - 0.274 * (imgDist(:,:,2)) - 0.322 * (imgDist(:,:,3));
        imgRefQ = 0.211 * (imgRef(:,:,1)) - 0.523 * (imgRef(:,:,2)) + 0.312 * (imgRef(:,:,3));
        imgDistQ = 0.211 * (imgDist(:,:,1)) - 0.523 * (imgDist(:,:,2)) + 0.312 * (imgDist(:,:,3));
    else
        imgRefY = double(imgRef);
        imgDistY = double(imgDist);
    end
       
    %% subsampling    
    if preprocessWithSubsampling
        imgRefY = HaarPSISubsample(imgRefY,boundaryTreatment);
        imgDistY = HaarPSISubsample(imgDistY,boundaryTreatment);    
        if colorImage
            imgRefQ = HaarPSISubsample(imgRefQ,boundaryTreatment);
            imgDistQ = HaarPSISubsample(imgDistQ,boundaryTreatment);
            imgRefI = HaarPSISubsample(imgRefI,boundaryTreatment);
            imgDistI = HaarPSISubsample(imgDistI,boundaryTreatment);
        end 
    end
    
    %% pre-allocate variables
    if colorImage
        localSimilarities = zeros([size(imgRefY),3]);
        weights = zeros([size(imgRefY),3]);  
    else
        localSimilarities = zeros([size(imgRefY),2]);
        weights = zeros([size(imgRefY),2]);  
    end    
    
    %% Haar wavelet decomposition
    nScales = 3;
    coeffsRefY = HaarPSIDec(imgRefY,wavelet,boundaryTreatment,nScales);
    coeffsDistY = HaarPSIDec(imgDistY,wavelet,boundaryTreatment,nScales);    
    if colorImage
        coeffsRefQ = abs(imfilter(imgRefQ,ones(2,2)/4,'same','conv',boundaryTreatment));
        coeffsDistQ = abs(imfilter(imgDistQ,ones(2,2)/4,'same','conv',boundaryTreatment));
        coeffsRefI = abs(imfilter(imgRefI,ones(2,2)/4,'same','conv',boundaryTreatment));
        coeffsDistI = abs(imfilter(imgDistI,ones(2,2)/4,'same','conv',boundaryTreatment));
    end
    
    %% compute weights and similarity for each orientation
    for ori = 1:2        
        weights(:,:,ori) = max(abs(coeffsRefY(:,:,3 + (ori-1)*nScales)), abs(coeffsDistY(:,:,3 + (ori-1)*nScales)));
        coeffsRefYMag = abs(coeffsRefY(:,:,(1:2) + (ori-1)*nScales));
        coeffsDistYMag = abs(coeffsDistY(:,:,(1:2) + (ori-1)*nScales));
        localSimilarities(:,:,ori) = sum(((2*coeffsRefYMag.*coeffsDistYMag + C)./(coeffsRefYMag.^2 + coeffsDistYMag.^2 + C)),3)/2;    
    end
    
    %% compute similarities for color channels
    if colorImage         
        similarityI = ((2*coeffsRefI.*coeffsDistI + C) ./(coeffsRefI.^2 + coeffsDistI.^2 + C));
        similarityQ = ((2*coeffsRefQ.*coeffsDistQ + C) ./(coeffsRefQ.^2 + coeffsDistQ.^2 + C));
        localSimilarities(:,:,3) = ((similarityI)+(similarityQ))/2;
        weights(:,:,3) = (weights(:,:,1) + weights(:,:,2))/2;
    end
    
    %% compute final score
    similarity = HaarPSILogInv(sum((HaarPSILog(localSimilarities(:),alpha)).*weights(:))/sum(weights(:)),alpha)^2;
    
    %% output maps
    if nargout > 1
        similarityMaps = localSimilarities;
    end
    if nargout > 2
        weightMaps = weights;
    end
end

function coeffs = HaarPSIDec(X,wavelet,boundaryTreatment,nScales)
    [h,g] = wfilters(wavelet);
    coeffs = zeros([size(X),2*nScales]);    
    hnew = h;
    hnew2 = hnew;
    g2 = g;
    for k = 1:nScales
        waveletFilter = hnew2'*g2;
        coeffs(:,:,k) = imfilter(X,waveletFilter,'same','conv',boundaryTreatment);
        coeffs(:,:,k + nScales) = imfilter(X,waveletFilter','same','conv',boundaryTreatment);
        g = conv(h,dyadup(g));
        hnew = conv(h,dyadup(hnew));        
        g2 = g((find(abs(g)>0,1,'first')):(find(abs(g)>0,1,'last')));
        hnew2 = hnew((find(abs(hnew)>0,1,'first')):(find(abs(hnew)>0,1,'last')));
    end   
end

function imgSubsampled = HaarPSISubsample(img,boundaryTreatment)
    imgSubsampled = imfilter(img, ones(2,2)/4,'same','conv',boundaryTreatment);        
    imgSubsampled = imgSubsampled(1:2:end,1:2:end);
end

function val = HaarPSILog(x,alpha)
    val = 1./(1 + exp(-alpha.*(x)));
end

function val = HaarPSILogInv(x,alpha)
    val = log(x./(1-x))./alpha;
end

%  Written by Rafael Reisenhofer
%  Built on 08/05/2017