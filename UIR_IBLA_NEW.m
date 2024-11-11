clear all; close all; clc

% 参数定义
beta = [1/7; 1/25 ; 1/28];
t1 = 1;
t0 = 0.1;
D = 8;
B_thr_high = 0.5;
r_thr_high = 0.1;
as = 32;
win = 7;
use_lap = 1;
wl = [620, 540, 450];
Strch = @(x) (x - min(x(:))) .* (1 / (max(x(:)) - min(x(:))));

% 设置输入和输出文件夹
input_folder = uigetdir('', '选择包含图像的文件夹');
output_folder = uigetdir('', '选择保存处理后图像的文件夹');
if isequal(input_folder, 0) || isequal(output_folder, 0)
    disp('未选择文件夹');
    return;
end

% 获取文件夹中所有图片文件
file_list = dir(fullfile(input_folder, '*.bmp'));
file_list = [file_list; dir(fullfile(input_folder, '*.jpg'))];
file_list = [file_list; dir(fullfile(input_folder, '*.png'))];
file_list = [file_list; dir(fullfile(input_folder, '*.BMP'))];
file_list = [file_list; dir(fullfile(input_folder, '*.JPG'))];
file_list = [file_list; dir(fullfile(input_folder, '*.PNG'))];

% 批量处理每张图片
for i = 1:length(file_list)
    % 读取图像
    file_name = file_list(i).name;
    full_file_path = fullfile(input_folder, file_name);
    I = im2double(imread(full_file_path));
    [height, width, ~] = size(I);

    % 设置 lambda 值
    if width * height < 180000
        lambda = 10e-6;
    else
        lambda = 10e-3;
    end

    % 估计透射率和模糊
    [trans_mip, trans_red] = estTransRed(I, win);
    t_b_est = estBlur(I, win);
    t_blur_est = laplacian_matting(t_b_est, I, lambda);

    % 估计背景光和距离
    B = estBacklight(I, t_blur_est);
    mean_b = mean(B);
    mean_r = mean(reshape(I(:,:,1), height * width, 1));

    alpha = sigmf(mean_b, [as, B_thr_high]);
    alpha2 = sigmf(mean_r, [as, r_thr_high]);
    t_pro_a_est = trans_mip * alpha + trans_red * (1 - alpha);
    t_strch_blur_est = Strch(t_b_est);

    if use_lap == 0
        t_pro_est = imguidedfilter(t_pro_a_est * alpha2 + t_strch_blur_est * (1 - alpha2), I, 'NeighborhoodSize', [win, win]);
    else
        t_pro_est = laplacian_matting(t_pro_a_est * alpha2 + t_strch_blur_est * (1 - alpha2), I, lambda);
    end

    % 计算距离映射和透射率
    BLmap = zeros(size(I));
    for ind = 1:3
        BLmap(:,:,ind) = B(ind) * ones(height, width);
    end
    diff_BL_I = abs(BLmap - I);
    [max_diff_bl_i, pos] = max(diff_BL_I(:));
    R = max_diff_bl_i / max(BLmap(pos), 1 - BLmap(pos));

    % 计算透射率和复原图像
    BL = max(B, 0.1);
    b_sf  = -0.00113 * wl + 1.62517;
    cg2cr = (b_sf(2) * BL(1)) / (b_sf(1) * BL(2));
    cb2cr = (b_sf(3) * BL(1)) / (b_sf(1) * BL(3));
    
    beta(2) = beta(1) * cg2cr;
    beta(3) = beta(1) * cb2cr;

    dist = D * (1 - t_pro_est + (1 - R));
    trans = zeros(size(I));
    for ind = 1:3
        trans(:,:,ind) = exp(-beta(ind) .* dist);
    end

    J_pro = zeros(size(I));
    for ind = 1:3
        J_pro(:,:,ind) = B(ind) + (I(:,:,ind) - B(ind)) ./ max(trans(:,:,ind), t0);
    end
    J_pro(J_pro < 0) = 0;
    J_pro(J_pro > 1) = 1;


    % 保存处理后的图像
    [~, name, ext] = fileparts(file_name);
    output_file_name = fullfile(output_folder, strcat(name, '_processed', ext));
    imwrite(J_pro, output_file_name);
end

disp('批处理完成');
