function ImageProcessingGUI()
   % 创建主界面
    fig = uifigure('Name', '图像处理系统', 'Position', [100, 100, 1200, 800]);

    % 添加网格布局
    grid = uigridlayout(fig, [5, 4]);
    grid.ColumnWidth = {'1x', '1x', '1x', '1x'};
    grid.RowHeight = {50, 50, 50, '3x', '3x'};

    % 文件加载按钮
    btnLoad = uibutton(grid, 'Text', '加载图像', 'ButtonPushedFcn', @loadImage);
    btnLoad.Layout.Row = 1;
    btnLoad.Layout.Column = 1;

    % 功能菜单
    ddFunction = uidropdown(grid, ...
        'Items', {'图像缩放', '图像旋转', '添加噪声', '滤波处理', ...
                  '直方图均衡化', '灰度增强(线性)', '灰度增强(对数变换)', '灰度增强(指数变换)'...
                  '边缘检测', '特征提取(LBP)', '特征提取(HOG)', '分类(深度学习)', '直方图匹配','目标提取'}, ...
        'Value', '图像缩放');
    ddFunction.Layout.Row = 1;
    ddFunction.Layout.Column = 2;

    % 执行操作按钮
    btnProcess = uibutton(grid, 'Text', '执行操作', 'ButtonPushedFcn', @processImage);
    btnProcess.Layout.Row = 2;
    btnProcess.Layout.Column = 6;

    % 边缘检测算子选择菜单
    ddEdgeOperator = uidropdown(grid, ...
        'Items', {'Robert 算子', 'Prewitt 算子', 'Sobel 算子', '拉普拉斯算子'}, ...
        'Value', 'Sobel 算子', ...
        'Editable', 'off');
    ddEdgeOperator.Layout.Row = 1;
    ddEdgeOperator.Layout.Column = 4;

    % 缩放比例输入框
    lblScale = uilabel(grid, 'Text', '缩放比例:');
    lblScale.Layout.Row = 2;
    lblScale.Layout.Column = 1;

    txtScale = uieditfield(grid, 'numeric', 'Value', 1.0);
    txtScale.Layout.Row = 2;
    txtScale.Layout.Column = 2;

    % 旋转角度输入框
    lblRotate = uilabel(grid, 'Text', '旋转角度 (°):');
    lblRotate.Layout.Row = 2;
    lblRotate.Layout.Column = 3;

    txtRotate = uieditfield(grid, 'numeric', 'Value', 0);
    txtRotate.Layout.Row = 2;
    txtRotate.Layout.Column = 4;

    % 噪声类型选择菜单
    lblNoise = uilabel(grid, 'Text', '噪声类型:');
    lblNoise.Layout.Row = 3;
    lblNoise.Layout.Column = 1;

    ddNoiseType = uidropdown(grid, ...
        'Items', {'高斯噪声', '椒盐噪声', '泊松噪声'}, ...
        'Value', '高斯噪声');
    ddNoiseType.Layout.Row = 3;
    ddNoiseType.Layout.Column = 2;

    % 滤波方式选择菜单
    lblFilter = uilabel(grid, 'Text', '滤波方式:');
    lblFilter.Layout.Row = 3;
    lblFilter.Layout.Column = 3;

    ddFilterType = uidropdown(grid, ...
        'Items', {'均值滤波', '高斯滤波'}, ...
        'Value', '均值滤波');
    ddFilterType.Layout.Row = 3;
    ddFilterType.Layout.Column = 4;

    % 原始图像面板
    panelOriginal = uipanel(grid, 'Title', '原始图像');
    panelOriginal.Layout.Row = 4;
    panelOriginal.Layout.Column = 1:2;
    axOriginal = uiaxes(panelOriginal);

    % 处理后图像面板
    panelProcessed = uipanel(grid, 'Title', '处理后图像');
    panelProcessed.Layout.Row = 4;
    panelProcessed.Layout.Column = 3:4;
    axProcessed = uiaxes(panelProcessed);

     % 灰度直方图面板
    panelHistogram = uipanel(grid, 'Title', '灰度直方图');
    panelHistogram.Layout.Row = 4;
    panelHistogram.Layout.Column = 5:6;
    axHistogram = uiaxes(panelHistogram);

    % 初始化全局变量
    global originalImage processedImage;
    originalImage = [];
    processedImage = [];

    % 加载图像的回调函数
    function loadImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件 (*.jpg, *.png, *.bmp)'});
        if isequal(file, 0)
            return;
        end
        imgPath = fullfile(path, file);
        originalImage = imread(imgPath);
        imshow(originalImage, 'Parent', axOriginal);
        title(axOriginal, '原始图像');
        processedImage = originalImage;
        imshow(processedImage, 'Parent', axProcessed);
        title(axProcessed, '处理后图像');
        showHistogram(originalImage);
    end

    % 显示直方图的函数
    function showHistogram(image)
        % 检查并转换为灰度图像
        if size(image, 3) == 3
            grayImage = rgb2gray(image);
        else
            grayImage = image;
        end

        % 初始化灰度计数数组
        grayLevels = 0:255;        % 灰度级范围
        pixelCount = zeros(1, 256); % 初始化每个灰度值的计数

        % 手动统计每个灰度级的像素数量
        for i = 1:256
            pixelCount(i) = sum(grayImage(:) == grayLevels(i));
        end

        % 绘制直方图
        bar(axHistogram, grayLevels, pixelCount, 'BarWidth', 1, 'FaceColor', 'b');
        xlim(axHistogram, [0 255]); % 灰度范围
        ylim(axHistogram, [0 max(pixelCount) * 1.1]); % 调整y轴范围，便于观察
        title(axHistogram, '灰度直方图');
        xlabel(axHistogram, '灰度级');
        ylabel(axHistogram, '像素数');
    end


    % 执行图像处理的回调函数
    function processImage(~, ~)
        if isempty(originalImage)
            uialert(fig, '请先加载图像！', '错误');
            return;
        end
        selectedFunction = ddFunction.Value;

        switch selectedFunction
            case '图像缩放'
                scaleFactor = txtScale.Value;
                processedImage = customResize(originalImage, scaleFactor);

            case '图像旋转'
                angle = txtRotate.Value;
                processedImage = customRotate(originalImage, angle);

            case '添加噪声'
                switch ddNoiseType.Value
                    case '高斯噪声'
                        processedImage = addGaussianNoise(originalImage, 0, 0.01); % 设定均值和方差
                    case '椒盐噪声'
                        processedImage = addSaltPepperNoise(originalImage, 0.05); % 设定噪声密度
                    case '泊松噪声'
                        processedImage = addPoissonNoise(originalImage);
                end


             case '滤波处理'
                % 获取滤波方式
                switch ddFilterType.Value
                    case '均值滤波'
                        % 设置滤波参数
                        window_size = 5;    % 滤波器窗口大小（奇数）
                        processedImage = mean_filter(originalImage, window_size);
                    case '高斯滤波'
                        % 设置滤波参数
                        window_size = 5;    % 滤波器窗口大小（奇数）
                        sigma = 1.5;        % 高斯滤波的标准差
                        processedImage = gaussian_filter(originalImage, window_size, sigma);

                end


            case '直方图均衡化'
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end
                processedImage = customHistogramEqualization(grayImage);


            case '直方图匹配'
                % 检查是否是灰度图像
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end

                % 加载参考图像
                refImage = imread('bird.jpg');
                if size(refImage, 3) == 3
                    refGrayImage = rgb2gray(refImage);
                else
                    refGrayImage = refImage;
                end

                % 调用直方图匹配函数
                processedImage = customHistogramMatching(grayImage, refGrayImage);



            case '灰度增强(线性)'
                a = 1.5; % 增强系数
                b = 20;  % 偏移量
                processedImage = linearContrastEnhancement(rgb2gray(originalImage),a,b);
            case '灰度增强(对数变换)'
                gamma = 1; % 增强参数
                processedImage = logarithmicContrastEnhancement(rgb2gray(originalImage),gamma);
            case '灰度增强(指数变换)'
                gamma = 0.5; % 增强参数
                processedImage = exponentialContrastEnhancement(rgb2gray(originalImage),gamma);
            case '边缘检测'
                operator = ddEdgeOperator.Value;
                switch operator
                    case 'Robert 算子'
                        processedImage = applyRobertEdgeDetection(originalImage);
                    case 'Prewitt 算子'
                        processedImage = applyPrewittEdgeDetection(originalImage);
                    case 'Sobel 算子'
                        processedImage = applySobelEdgeDetection(originalImage);
                    case '拉普拉斯算子'
                        processedImage = applyLaplacianEdgeDetection(originalImage);
                end

            case '特征提取(LBP)'
                % 检查是否加载了图像
                if isempty(originalImage)
                    uialert(fig, '请先加载图像！', '错误');
                    return;
                end
                % 转换为灰度图像
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end

                % 提取 LBP 特征
                radius = 1; % 半径
                numPoints = 8; % 邻域点数
                [lbpFeatures, processedImage] = customLBP(grayImage, radius, numPoints);

                % 显示 LBP 图像到处理后图像坐标轴
                axes(axProcessed); % 设置当前坐标轴为 axProcessed
                cla(axProcessed); % 清空当前内容

                % 确保 lbpImage 是有效的图像数据
                if ~isempty(processedImage) && isnumeric(processedImage)
                    disp(size(processedImage));  % 输出图像的大小，检查其是否正确
                    imshow(processedImage, [], 'Parent', axProcessed); % 使用 imshow 显示特征图
                    title(axProcessed, 'LBP 特征图像');
                else
                    uialert(fig, 'LBP 特征图像无效或为空', '错误');
                end

                disp('LBP 特征提取完成');



            case '特征提取(HOG)'
                % 转换为灰度图像
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end

                % 生成初步掩膜（可以基于先验或程序生成）
                [height, width] = size(grayImage);
                fgMask = false(height, width);
                bgMask = false(height, width);

                % 自动生成前景掩膜（假设中心为前景）
                fgMask(round(height/4):round(3*height/4), round(width/4):round(3*width/4)) = true;

                % 自动生成背景掩膜（假设边缘为背景）
                bgMask(1:round(height/10), :) = true; % 上边缘
                bgMask(round(9*height/10):end, :) = true; % 下边缘
                bgMask(:, 1:round(width/10)) = true; % 左边缘
                bgMask(:, round(9*width/10):end) = true; % 右边缘

                % 使用 `lazysnapping` 函数进行分割
                L = lazysnapping(originalImage, bgMask, fgMask);

                % 将分割结果转换为掩膜
                segmentedMask = L == 1;

                % 应用掩膜到原图
                processedImage = bsxfun(@times, originalImage, cast(segmentedMask, 'like', originalImage));

                % 显示分割结果
                imshow(processedImage, 'Parent', axProcessed);




            case '分类(深度学习)'
                if isempty(svmModel)
                    % 如果模型未加载，则先训练模型
                    uialert(fig, '正在训练模型，请稍等...', '提示');
                    [svmModel, categories] = trainHOGSVM();
                    uialert(fig, '模型训练完成！', '提示');
                end
                % 使用已加载的模型进行分类
                grayImage = rgb2gray(originalImage);
                resizedImage = imresize(grayImage, [64, 64]); % 调整图像大小
                featureVector = extractHOGFeatures(resizedImage); % 提取HOG特征
                predictedLabel = predict(svmModel, featureVector);
                categoryName = categories(predictedLabel).name;
                uialert(fig, ['分类结果: ', categoryName], '分类完成');
                return;
        end

        % 显示处理后的图像
        imshow(processedImage, 'Parent', axProcessed);
        title(axProcessed, '处理后图像');
    end

    % 训练模型的回调函数
    function trainModel(~, ~)
        uialert(fig, '正在训练模型，请稍候...', '提示');
        [svmModel, categories] = trainHOGSVM();
        uialert(fig, '模型训练完成！可以进行分类操作。', '提示');
    end

    % HOG + SVM训练函数
    function [svmModel, categories] = trainHOGSVM()
        % 数据集路径
        datasetPath = 'C:\Users\admin\Downloads\CUB_200_2011.zip\CUB_200_2011\images';
        categories = dir(datasetPath);
        categories = categories([categories.isdir]); % 只保留文件夹
        categories = categories(3:end); % 排除'.'和'..'

        features = [];
        labels = [];

        for i = 1:length(categories)
            categoryName = categories(i).name;
            categoryPath = fullfile(datasetPath, categoryName);
            imageFiles = dir(fullfile(categoryPath, '*.jpg'));
            for j = 1:min(50, length(imageFiles))
                imgPath = fullfile(categoryPath, imageFiles(j).name);
                img = imread(imgPath);
                if size(img, 3) == 3
                    img = rgb2gray(img);
                end
                img = imresize(img, [64, 64]);
                hogFeature = extractHOGFeatures(img);
                features = [features; hogFeature];
                labels = [labels; i];
            end
        end
        svmModel = fitcecoc(features, labels);
    end
end



function enhancedImage = linearContrastEnhancement(grayImage, a, b)
    % 进行线性变换增强对比度
    enhancedImage = a * double(grayImage) + b;
    
    % 确保输出值在有效的图像范围内 [0, 255]
    enhancedImage = uint8(min(max(enhancedImage, 0), 255));
end

% 非线性对比度增强：对数变换
function logEnhancedImage = logarithmicContrastEnhancement(grayImage, c)
    % 对数变换增强对比度
    logEnhancedImage = c * log(1 + double(grayImage));
    
    % 将输出归一化到 [0, 255]
    logEnhancedImage = uint8(255 * mat2gray(logEnhancedImage));
end


% 非线性对比度增强：指数变换
function expEnhancedImage = exponentialContrastEnhancement(grayImage, c)
    % 归一化图像到 [0, 1]
    normalizedImage = double(grayImage) / 255;

    % 应用指数变换
    expTransformed = c * (exp(normalizedImage) - 1);

    % 归一化结果到 [0, 1]
    expTransformed = expTransformed / max(expTransformed(:));

    % 映射回 [0, 255] 并转换为 uint8
    expEnhancedImage = uint8(expTransformed * 255);
end



% Robert 算子进行边缘检测
function edges_robert = applyRobertEdgeDetection(inputImage)
    % 检查输入图像是否为灰度图像
    if size(inputImage, 3) == 3
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage;
    end

    % 定义 Robert 算子
    Gx = [1 0; 0 -1];
    Gy = [0 1; -1 0];

    % 卷积计算梯度
    gradientX = conv2(double(grayImage), Gx, 'same');
    gradientY = conv2(double(grayImage), Gy, 'same');

    % 计算梯度幅值
    edges_robert = sqrt(gradientX.^2 + gradientY.^2);

    % 标准化到 [0, 255]
    edges_robert = uint8(255 * mat2gray(edges_robert));
end




% Prewitt算子进行边缘检测
function edges_prewitt = applyPrewittEdgeDetection(inputImage)
    if size(inputImage, 3) == 3
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage;
    end

    % 定义 Prewitt 算子
    Gx = [-1 0 1; -1 0 1; -1 0 1];
    Gy = [-1 -1 -1; 0 0 0; 1 1 1];

    % 卷积计算梯度
    gradientX = conv2(double(grayImage), Gx, 'same');
    gradientY = conv2(double(grayImage), Gy, 'same');

    % 计算梯度幅值
    edges_prewitt = sqrt(gradientX.^2 + gradientY.^2);

    % 标准化到 [0, 255]
    edges_prewitt = uint8(255 * mat2gray(edges_prewitt));
end



% Sobel算子进行边缘检测
function edges_sobel = applySobelEdgeDetection(inputImage)
    if size(inputImage, 3) == 3
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage;
    end

    % 定义 Sobel 算子
    Gx = [-1 0 1; -2 0 2; -1 0 1];
    Gy = [-1 -2 -1; 0 0 0; 1 2 1];

    % 卷积计算梯度
    gradientX = conv2(double(grayImage), Gx, 'same');
    gradientY = conv2(double(grayImage), Gy, 'same');

    % 计算梯度幅值
    edges_sobel = sqrt(gradientX.^2 + gradientY.^2);

    % 标准化到 [0, 255]
    edges_sobel = uint8(255 * mat2gray(edges_sobel));
end


%拉普拉斯算子进行边缘检测
function edges_laplacian = applyLaplacianEdgeDetection(inputImage)
    if size(inputImage, 3) == 3
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage;
    end

    % 定义拉普拉斯算子
    LaplacianKernel = [0 -1 0; -1 4 -1; 0 -1 0];

    % 卷积计算
    edges_laplacian = conv2(double(grayImage), LaplacianKernel, 'same');

    % 取绝对值并标准化到 [0, 255]
    edges_laplacian = uint8(255 * mat2gray(abs(edges_laplacian)));
end


% 膨胀
function dilatedImage = customDilation(binaryImage, structuringElement)
    % 获取图像和结构元素的尺寸
    [rows, cols] = size(binaryImage);
    [seRows, seCols] = size(structuringElement);

    % 计算结构元素的中心
    seCenterRow = floor(seRows / 2) + 1;
    seCenterCol = floor(seCols / 2) + 1;

    % 初始化膨胀结果图像
    dilatedImage = zeros(rows, cols);

    % 遍历图像的每个像素
    for i = 1:rows
        for j = 1:cols
            % 遍历结构元素
            for m = 1:seRows
                for n = 1:seCols
                    % 计算结构元素对应的图像位置
                    ii = i + m - seCenterRow;
                    jj = j + n - seCenterCol;

                    % 判断是否越界
                    if ii > 0 && ii <= rows && jj > 0 && jj <= cols
                        % 如果结构元素和原图像重叠区域有1，则膨胀结果为1
                        if structuringElement(m, n) && binaryImage(ii, jj)
                            dilatedImage(i, j) = 1;
                        end
                    end
                end
            end
        end
    end
end

% 连通区域填充
function filledImage = customFillHoles(binaryImage)
    % 获取图像尺寸
    [rows, cols] = size(binaryImage);

    % 创建初始标记图像（边界为1，内部为0）
    filledImage = binaryImage;
    marker = zeros(rows, cols);
    marker([1, rows], :) = 1;  % 设置上下边界
    marker(:, [1, cols]) = 1;  % 设置左右边界

    % 迭代填充
    prevMarker = marker;
    while true
        % 膨胀操作
        marker = customDilation(marker, ones(3, 3));
        % 限制在原图像的补集中
        marker = marker & ~binaryImage;

        % 如果标记图像没有变化，则停止
        if isequal(marker, prevMarker)
            break;
        end
        prevMarker = marker;
    end

    % 填充孔洞
    filledImage = ~marker | binaryImage;
end



%目标提取
function targetImage = extractTargetFromEdges(edgeImage)
    % 检查输入是否为二值图像
    if ~islogical(edgeImage)
        edgeImage = edgeImage > 0;  % 转换为二值图像
    end

    % 自定义膨胀操作
    structuringElement = ones(5);  % 等效于 strel('disk', 5)
    dilatedImage = customDilation(edgeImage, structuringElement);

    % 自定义孔洞填充
    targetImage = customFillHoles(dilatedImage);
end



function [lbpFeatures, lbpImage] = customLBP(grayImage, radius, numPoints)
    [rows, cols] = size(grayImage);
    lbpImage = zeros(rows, cols); % 初始化 LBP 图像
    angles = linspace(0, 2*pi, numPoints+1);
    angles(end) = [];
    xOffsets = radius * cos(angles);
    yOffsets = radius * sin(angles);

    for r = radius+1 : rows-radius
        for c = radius+1 : cols-radius
            centerPixel = grayImage(r, c);
            neighbors = zeros(1, numPoints);

            for i = 1:numPoints
                neighborPixel = grayImage(round(r + yOffsets(i)), round(c + xOffsets(i)));
                neighbors(i) = neighborPixel > centerPixel;
            end

            % 将二进制模式转换为十进制值
            lbpImage(r, c) = sum(neighbors .* (2.^(0:numPoints-1)));
        end
    end

    % 计算直方图作为特征
    lbpFeatures = histcounts(lbpImage(:), 0:(2^numPoints)-1, 'Normalization', 'probability');
end


function equalizedImage = customHistogramEqualization(image)
    % 检查是否为灰度图像
    if size(image, 3) == 3
        grayImage = rgb2gray(image);
    else
        grayImage = image;
    end
    grayImage = double(grayImage);

    % 获取图像尺寸和像素总数
    [rows, cols] = size(grayImage);
    numPixels = rows * cols;

    % 计算直方图
    grayLevels = 0:255;  % 灰度级范围
    pixelCount = zeros(1, 256);
    for i = 1:256
        pixelCount(i) = sum(grayImage(:) == grayLevels(i));
    end

    % 计算累计分布函数 (CDF)
    cdf = cumsum(pixelCount) / numPixels;
    newGrayLevels = round(cdf * 255);

    % 创建映射后的图像
    equalizedImage = zeros(rows, cols);
    for i = 1:256
        equalizedImage(grayImage == grayLevels(i)) = newGrayLevels(i);
    end

    % 转换为 uint8 格式
    equalizedImage = uint8(equalizedImage);
end


function matchedImage = customHistogramMatching(inputImage, refImage)
    % 检查输入图像是否为灰度图像
    if size(inputImage, 3) == 3
        inputGray = rgb2gray(inputImage);
    else
        inputGray = inputImage;
    end

    if size(refImage, 3) == 3
        refGray = rgb2gray(refImage);
    else
        refGray = refImage;
    end
    % 计算输入图像的直方图和累计分布函数 (CDF)
    inputHist = histcounts(inputGray(:), 0:256); % 直方图
    inputCDF = cumsum(inputHist) / numel(inputGray); % 累计分布函数

    % 计算参考图像的直方图和累计分布函数 (CDF)
    refHist = histcounts(refGray(:), 0:256); % 直方图
    refCDF = cumsum(refHist) / numel(refGray); % 累计分布函数

    % 构造灰度值映射
    grayLevels = 0:255;
    mapping = zeros(256, 1);
    for g = 1:256
        [~, idx] = min(abs(inputCDF(g) - refCDF)); % 寻找最接近的参考 CDF 值
        mapping(g) = grayLevels(idx); % 映射到参考灰度值
    end
    % 应用灰度值映射
    matchedImage = mapping(double(inputGray) + 1);
    % 转换为 uint8 格式
    matchedImage = uint8(matchedImage);
end


% 图像缩放（双线性插值）
function scaledImage = customResize(inputImage, scaleFactor)
    [rows, cols, channels] = size(inputImage);
    newRows = round(rows * scaleFactor);
    newCols = round(cols * scaleFactor);

    % 创建新图像
    scaledImage = zeros(newRows, newCols, channels, 'like', inputImage);

    % 计算缩放比例
    rowScale = rows / newRows;
    colScale = cols / newCols;
    for r = 1:newRows
        for c = 1:newCols
            origR = r * rowScale;
            origC = c * colScale;

            % 获取周围像素
            r1 = floor(origR);
            r2 = min(r1 + 1, rows);
            c1 = floor(origC);
            c2 = min(c1 + 1, cols);
            % 计算权重
            deltaR = origR - r1;
            deltaC = origC - c1;
            % 双线性插值
            for ch = 1:channels
                scaledImage(r, c, ch) = ...
                    (1 - deltaR) * (1 - deltaC) * inputImage(r1, c1, ch) + ...
                    (1 - deltaR) * deltaC * inputImage(r1, c2, ch) + ...
                    deltaR * (1 - deltaC) * inputImage(r2, c1, ch) + ...
                    deltaR * deltaC * inputImage(r2, c2, ch);
            end
        end
    end
end

% 图像旋转（最近邻插值）
function rotatedImage = customRotate(inputImage, angle)
    angleRad = deg2rad(angle);
    [rows, cols, channels] = size(inputImage);

    % 计算新图像尺寸
    diagLength = ceil(sqrt(rows^2 + cols^2));
    newRows = diagLength;
    newCols = diagLength;
    rotatedImage = zeros(newRows, newCols, channels, 'like', inputImage);

    % 原图像中心
    centerX = cols / 2;
    centerY = rows / 2;

    % 新图像中心
    newCenterX = newCols / 2;
    newCenterY = newRows / 2;

    for r = 1:newRows
        for c = 1:newCols
            % 计算反向映射
            x = (c - newCenterX) * cos(angleRad) + (r - newCenterY) * sin(angleRad) + centerX;
            y = -(c - newCenterX) * sin(angleRad) + (r - newCenterY) * cos(angleRad) + centerY;

            % 最近邻插值
            x = round(x);
            y = round(y);

            if x >= 1 && x <= cols && y >= 1 && y <= rows
                rotatedImage(r, c, :) = inputImage(y, x, :);
            end
        end
    end
end


% 高斯噪声
function noisyImage = addGaussianNoise(inputImage, mean, variance)
    noise = mean + sqrt(variance) * randn(size(inputImage));
    noisyImage = double(inputImage) + noise;
    noisyImage = uint8(max(0, min(255, noisyImage)));
end

% 椒盐噪声
function noisyImage = addSaltPepperNoise(inputImage, noiseDensity)
    noisyImage = inputImage;
    numPixels = numel(inputImage);
    numSalt = round(noiseDensity * numPixels / 2);
    numPepper = numSalt;

    % 添加盐噪声
    saltIndices = randperm(numPixels, numSalt);
    noisyImage(saltIndices) = 255;

    % 添加胡椒噪声
    pepperIndices = randperm(numPixels, numPepper);
    noisyImage(pepperIndices) = 0;
end

% 泊松噪声
function noisyImage = addPoissonNoise(inputImage)
    % 确保图像为灰度图像
    if size(inputImage, 3) == 3
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage;
    end
    % 将灰度图像转换为 double 类型，归一化到 [0, 1]
    normalizedImage = double(grayImage) / 255;
    % 生成泊松噪声
    poissonNoise = poissrnd(normalizedImage);
    % 重新归一化到 [0, 255]
    noisyImage = uint8(poissonNoise * 255);
end



function segmentedImage = lazySnapping(inputImage, fgMask, bgMask)
    % 检查输入图像是否为彩色
    if size(inputImage, 3) == 3
        labImage = rgb2lab(inputImage); % 转换为 LAB 颜色空间
    else
        error('Lazy Snapping requires a color input image.');
    end

    [height, width, ~] = size(inputImage);

    % 初始化图的邻接矩阵
    numNodes = height * width;
    graphWeights = sparse(numNodes, numNodes);

    % 计算权重（基于颜色差异和位置邻接）
    scalingFactor = 10; % 调整权重的缩放因子
    for y = 1:height - 1
        for x = 1:width - 1
            currentIdx = (y - 1) * width + x;

            % 像素向右邻居的权重
            rightIdx = currentIdx + 1;
            diffRight = squeeze(labImage(y, x, :) - labImage(y, x + 1, :));
            weightRight = exp(-norm(diffRight)^2 / scalingFactor);
            graphWeights(currentIdx, rightIdx) = weightRight;
            graphWeights(rightIdx, currentIdx) = weightRight;

            % 像素向下邻居的权重
            bottomIdx = currentIdx + width;
            diffBottom = squeeze(labImage(y, x, :) - labImage(y + 1, x, :));
            weightBottom = exp(-norm(diffBottom)^2 / scalingFactor);
            graphWeights(currentIdx, bottomIdx) = weightBottom;
            graphWeights(bottomIdx, currentIdx) = weightBottom;
        end
    end

    % 构建图割算法的终端权重（前景/背景）
    fgWeights = zeros(numNodes, 1);
    bgWeights = zeros(numNodes, 1);

    fgPixels = find(fgMask);
    bgPixels = find(bgMask);

    fgWeights(fgPixels) = Inf; % 确保前景像素属于前景
    bgWeights(bgPixels) = Inf; % 确保背景像素属于背景

    % 使用图割算法进行分割
    [~, labels] = graphmincut(graphWeights, fgWeights, bgWeights);

    % 重建分割图像
    segmentedImage = reshape(labels, [height, width]);

    % 可视化分割结果
    figure; imshow(segmentedImage, []);
    title('Segmented Image');
end


function output = mean_filter(input_image, window_size)
    [height, width, channels] = size(input_image);
   
    output = zeros(height, width, channels, 'uint8');
    
    half_window = floor(window_size / 2);
    
    % 遍历每个像素并应用均值滤波
    for i = 1:height
        for j = 1:width
            % 定义当前像素的邻域范围
            x_min = max(1, i - half_window);
            x_max = min(height, i + half_window);
            y_min = max(1, j - half_window);
            y_max = min(width, j + half_window);
            
            % 提取邻域区域
            region = input_image(x_min:x_max, y_min:y_max, :);
            
            % 计算邻域区域的均值
            output(i, j, :) = mean(mean(region, 1), 2);
        end
    end
end


function output = gaussian_filter(input_image, window_size, sigma)
    [height, width, channels] = size(input_image);
    output = zeros(height, width, channels, 'uint8');
    % 创建高斯核
    [X, Y] = meshgrid(-floor(window_size / 2):floor(window_size / 2), ...
                      -floor(window_size / 2):floor(window_size / 2));
    gaussian_kernel = exp(-(X.^2 + Y.^2) / (2 * sigma^2));
    gaussian_kernel = gaussian_kernel / sum(gaussian_kernel(:));  % 归一化
    % 遍历每个像素并应用高斯滤波
    for i = 1:height
        for j = 1:width
            % 定义当前像素的邻域范围
            x_min = max(1, i - floor(window_size / 2));
            x_max = min(height, i + floor(window_size / 2));
            y_min = max(1, j - floor(window_size / 2));
            y_max = min(width, j + floor(window_size / 2));
            
            % 提取邻域区域
            region = input_image(x_min:x_max, y_min:y_max, :);
            
            % 计算加权平均（使用高斯核）
            for c = 1:channels
                output(i, j, c) = sum(sum(double(region(:,:,c)) .* gaussian_kernel(1:size(region,1), 1:size(region,2)))) / sum(gaussian_kernel(:));
            end
        end
    end
end





