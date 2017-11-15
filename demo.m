flowfileName = 'example/flow10.flo';
I0 = double(imread('example/frame10.png'));
I1 = double(imread('example/frame11.png'));

%MPI-Sintel flow read function
u0 = flow_read(flowfileName);

interpFrame = interp_frame(I0, I1, u0, 0.5, true);

% read ground truth image and calculate psnr
Igt = imread('example/frame10i11.png');
figure
imshow(interpFrame);
title(['Interpolated frame, psnr: ', num2str(psnr(Igt, interpFrame))]);

