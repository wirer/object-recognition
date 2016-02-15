% face recognition using ADABOOST
clear;
debug = 1; % switch the debug mode off (0) or on (1), (2) for verbose mode

face_path = './img_database/faces/';
bg_path = './img_database/background/';
face_files = dir([face_path '*.jpg']);
bg_files = dir([bg_path '*.jpg']);

if debug
    num_img = 1;
else
    num_img = 450;
end

% set up the filters
% gabor in 4 orientations
gabor(4).even = 0;
for i=1:4
    [gabor(i).even gabor(i).odd] = GaborD(5, 15, 15, i*pi/4, 2, 0, 0); % Apply Gabor function
    gabor(i).even = gabor(i).even-mean(mean(gabor(i).even));
    gabor(i).odd = gabor(i).odd-mean(mean(gabor(i).odd));
    
    if debug > 1 % show figures in verbose debug mode
        figure('name','gabor filter g1 (even)'), imagesc(gabor(i).even); colormap(gray);    % Show Gabor function
        figure('name','gabor filter g2 (odd)'), imagesc(gabor(i).odd); colormap(gray);     % Show Gabor function
    end
end
% gaussian filter for the pyramid
gauss = gauss2d ([3 3], 1, [2 2]);
if debug > 1
    figure('name', 'gaussian filter'), imagesc(gauss); colormap(gray);
end

%% load images into memory structure
images(2*num_img)= struct('data', 0, 'path', '', 'hasFace', 0, 'features', [], 'inTrainingSet', []);
for i = 1 : 2*num_img
    if i <= num_img
        images(i).path = [face_path, face_files(i).name];
        images(i).hasFace = 1;
    else
        images(i).path = [bg_path, bg_files(i-num_img).name];
        images(i).hasFace = -1;
    end
    images(i).inTrainingSet = -1; % 1 means the image is in the training set
    
    img_data = imread(images(i).path,'JPG');
   if ndims(img_data) >= 3 || size(img_data, 3) == 3
       img_data = rgb2gray(img_data);
   end
    img_data = imresize(img_data, [160 240]);
    
    images(i).data = uint8((double(img_data)-double(min(min(img_data))))* ...
        (255/double(max(max(img_data))-min(min(img_data))))); % normalize

    if debug > 1
        figure('name', 'input file'), imagesc(images(i).data); colormap(gray);
    end
end


%% set up the training set
for nn=1:2*num_img
    img_data =  images(nn).data;
    for i=1:4
        conv_even = conv2(double(img_data),gabor(i).even,'same');   % 2D convolution
        conv_odd = conv2(double(img_data), gabor(i).odd, 'same');
        conv_images = conv_even.^2 + conv_odd.^2;
        if debug > 1
            figure('name', 'convolved image'), imagesc(conv_images); colormap(gray);    % Show convolved images
        end
        
        % combine all images (add them)
        if i==1
            combine = conv_images;
        else
        combine = combine + conv_images;
      end
    end
    img_data = uint8((double(combine)-double(min(min(combine))))*(255/double(max(max(combine))-min(min(combine))))); % normalize

    if debug > 1
        figure('name', 'combined to power of gabor'), imagesc(img_data); colormap(gray);    % Show convolved images
    end
    
    % [img_o img_bc] = CORF(img_data,2.5,0.3);
    % figure('name', 'CORF O', 'NumberTitle', 'on'), imagesc(img_o); colormap(gray);    % Show convolved images
    % figure('name', 'CORF BC', 'NumberTitle', 'on'), imagesc(img_bc); colormap(gray);    % Show convolved images

    % Gaussian pyramid
    siz_pic = int16(size(img_data));
    for i=1:4
        in_img = conv2(double(img_data),gauss,'same');
        siz_pic = siz_pic./2;

        img_data = zeros(siz_pic(1), siz_pic(2));
        for ii=1:1:siz_pic(1)
            for j=1:1:siz_pic(2)
                img_data(ii, j) = in_img(ii*2, j*2);
            end
        end
        if debug > 1
            figure('name', 'pyramid (before resize)'); imagesc(in_img); colormap(gray);
            figure('name', 'pyramid'); imagesc(img_data); colormap(gray);
        end
    end
    images(nn).features = reshape(img_data, 150, 1);
end