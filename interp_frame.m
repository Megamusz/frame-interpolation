
function interp = interp_frame(I0, I1, u0, t, occ_detect)
% Interplate a frame based on previous(I0)/next frame(I1) and optical flow from I0 to I1
% The interplation algorithm is based on "A Database and Evaluation
% Methodology for Optical Flow"
% 
% Usage:
% interp = interp_frame(PREVIOUS_FRAME, NEXT_FRAME, OPTICAL_FLOW)
% interp = interp_frame(PREVIOUS_FRAME, NEXT_FRAME, OPTICAL_FLOW, t)
%
% PREVIOUS_FREAME/NEXT_FRAME is 3-channel rgb or grayscale images 
% OPTICAL_FLOW is a 2-channel, with first/second channel storing mvx/mvy 
% the size of OPTICAL_FLOW should match with I0/I1
% t is temporal distance t is in the range of (0, 1), default value is 0.5
% 


if nargin < 4
    t = 0.5; %temporal distance t is in the range of (0, 1)
end

if nargin < 5
    occ_detect = false;
end
% check input size
assert((size(I0, 1) == size(I1, 1)) && (size(I0, 2) == size(I1, 2)) && (size(I0, 1) == size(u0, 1)) && (size(I0, 2) == size(u0, 2)), 'I0/I1/u0 must have same cols&rows');


% Fc = flowToColor(u0);
% imshow(Fc);
% title('Original flow');
% imwrite(Fc, 'Original flow.png');

% Step 1. 
% Take the flow from I0 to I1 and forward warp (or splat)
% each flow value to the nearest destination pixel:

ut = nan(size(u0));
width = size(u0, 2);
height = size(u0, 1);

[xx, yy] = meshgrid(1:width, 1:height);
xt = round(xx + t*u0(:,:,1));
yt = round(yy + t*u0(:,:,2));

% warpingCount = zeros(height, width);

if occ_detect
    similarity = Inf(size(u0));
end
% TODO simplify the following code
for j = 1:height
    for i = 1:width
        j1 = yt(j, i);
        i1 = xt(j, i);
        if(i1 >=1 && i1 <= width && j1>=1 && j1<= height)
         
            if(occ_detect)
                e = (I1(j1, i1, :) - I0(j, i, :)).^2;
                s = sum(e(:));
                if( s < similarity(j1, i1) )
                    ut(j1, i1, :) = u0(j, i, :);
                    similarity(j1, i1) = s;
                end
            else
                ut(j1, i1, :) = u0(j, i, :);
            end
%             warpingCount(j1, i1) = warpingCount(j1, i1) + 1;
        end
    end
end

% occMask = warpingCount > 1;
% missingMask = warpingCount == 0;
% 
% figure;
% imshow([occMask missingMask]);
% title('occMask/missingMask');
% 
% utc = flowToColor(ut);
% figure;
% imshow(utc)
% imwrite(utc, 'Foward warping flow.png');
% title('Foward warping flow');


% Step 2. 
% Fill in any holes in the extrapolated motion field ut.
% (We use a simple outside-in filling strategy.)
uti = outside_in_fill(ut);
% uti = scanline_in_fill(ut, 1);
% uti = ut;


% figure;
% imshow(flowToColor(uti));
% imwrite(flowToColor(uti, 17.6), 'Interpolated flow.png');
% title('Interpolated flow');

% Step 3.
% Fetch the corresponding intensity values from both the
% first and second image and blend them together
% It(x) = (1?t)*I0(x?t*ut(x))+t*I1(x+(1?t)*ut(x)).
xt0 = max(1, min(width, xx - t*uti(:,:,1)));
yt0 = max(1, min(height, yy - t*uti(:,:,2)));

xt1 = max(1, min(width, xx + (1-t)*uti(:,:,1)));
yt1 = max(1, min(height, yy + (1-t)*uti(:,:,2)));

It = uint8(zeros(size(I0)));

% It0 = uint8(zeros(size(I0)));
% It1 = uint8(zeros(size(I0)));

for c = 1:size(I0, 3)
%     It0(:,:,c) = interp2(I0(:, :, c), xt0, yt0);
%     It1(:,:,c) = interp2(I1(:, :, c), xt1, yt1);
    It(:,:,c) = uint8((1-t)* interp2(I0(:, :, c), xt0, yt0) + t*interp2(I1(:, :, c), xt1, yt1));
end

interp = It;

% nanPos = isnan(uti);
% nanPos = repmat(nanPos, [1, 1, 2]);
% nanPos = nanPos(:, :, 1:3);
% It(nanPos) = I1(nanPos);
% 
% occPos = repmat(occMask, [1, 1, 3]);
% It(occPos) = I0(occPos);

% figure;
% imshow(uint8(It));
% Igt = imread('frame10i11.png');
% 
% title(['Interpolated frame, psnr: ', num2str(psnr(Igt, It))]);
% imwrite(It, 'Interpolated frame.png');

% figure;
% imshow(It0);
% figure;
% imshow(It1);

% err = Igt - It;
% figure;
% imshow(uint8(err.^2));
% title('error map');
% imwrite(uint8(err.^2), 'error map.png');
end

% the outside-in filling algorithm descriped in the paper. 
function output = outside_in_fill(input)

    rows = size(input, 1);
    cols = size(input, 2);
    
    cstart = 1;
    cend = cols;
    rstart = 1;
    rend = rows;
    lastValid = nan(2, 1);
    
    while(cstart < cend || rstart <rend)
        %top row
        for c = cstart:cend
            if(~isnan(input(rstart, c)))
                lastValid = input(rstart, c, :);
            else
                input(rstart, c, :) = lastValid;
            end
        end
        
        %right-most column
        for r = rstart:rend
            if(~isnan(input(r, cend)))
                lastValid = input(r, cend, :);
            else
                input(r, cend, :) = lastValid;
            end
        end
        
        %bottom row
        for c = cend:-1:cstart
            if(~isnan(input(rend, c)))
                lastValid = input(rend, c, :);
            else
                input(rend, c, :) = lastValid;
            end
        end
        
        %left-most column
        for r = rend:-1:rstart
            if(~isnan(input(r, cstart)))
                lastValid = input(r, cstart, :);
            else
                input(r, cstart, :) = lastValid;
            end
        end
        
        if(cstart < cend)
            cstart = cstart + 1;
            cend = cend - 1;
        end
        
        if(rstart < rend)
            rstart = rstart + 1;
            rend = rend - 1;
        end
    end
    
    output = input;
end

% the hole filling algorithm from KITTI benchmark
function output = scanline_in_fill(input, useVecMagCmp)
    height = size(input, 1);
    width = size(input, 2);
    
    if nargin < 2
        useVecMagCmp = 0;
    end

    for v=1:height
        count = 0;
        for u=1:width
            if(~isnan(input(v, u)))
                if(count >=1)
                    u1 = u-count;
                    u2 = u-1;
                    if(u1 > 1 && u2 <width)
                        if(useVecMagCmp)
                            fn1 = norm(reshape(input(v, u1-1, :), 2, 1), 1);
                            fn2 = norm(reshape(input(v, u2+1, :), 2, 1), 1);
                            interp = input(v, u1-1,:);
                        
                            if(fn2 < fn1)
                                interp = input(v, u2+1, :);
                            end
                        else
                            fu_ipol = min(input(v, u1-1, 1), input(v, u2 + 1, 1));
                            fv_ipol = min(input(v, u1-1, 2), input(v, u2 + 1, 2));
                        end
                        
                        for u_curr = u1:u2
                            if(useVecMagCmp)
                                input(v, u_curr, :) = interp;
                            else
                                input(v, u_curr, 1) = fu_ipol;
                                input(v, u_curr, 2) = fv_ipol;
                            
                            end
                        end
                    end
                end
                
                count = 0;
            else
                count = count + 1;
            end
        end
        %extrapolate to the left
        for u=1:width
            if ~isnan(input(v, u))
                for u2=1:u-1
                    input(v, u2, :) = input(v, u, :);
                end
                break;
            end
        end
        %extrapolate to the right
        for u=width:-1:1
            if ~isnan(input(v, u))
                for u2=u+1:width
                    input(v, u2, :) = input(v, u, :);
                end
                break;
            end
        end
    end
    
    
    for u= 1:width
        %extrapolate to the top
        for v=1:height
            if ~isnan(input(v, u))
                for v2=1:v-1
                    input(v2, u, :) = input(v, u, :);
                end
                break;
            end
        end
        %extrapolate to the bottom
        for v=height:-1:1
            if ~isnan(input(v, u))
                for v2=v+1:height
                    input(v2, u, :) = input(v, u, :);
                end
                break;
            end
        end
    end
    output = input;
end