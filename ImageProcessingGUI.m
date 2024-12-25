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
        'Items', {'均值滤波', '高斯滤波', '中值滤波'}, ...
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
                processedImage = imresize(originalImage, scaleFactor);

            case '图像旋转'
                angle = txtRotate.Value;
                processedImage = imrotate(originalImage, angle);

            case '添加噪声'
                switch ddNoiseType.Value
                    case '高斯噪声'
                        processedImage = imnoise(originalImage, 'gaussian');
                    case '椒盐噪声'
                        processedImage = imnoise(originalImage, 'salt & pepper');
                    case '泊松噪声'
                        processedImage = imnoise(originalImage, 'poisson');
                end

            case '滤波处理'
                switch ddFilterType.Value
                    case '均值滤波'
                        kernel = fspecial('average', [3 3]);
                        processedImage = imfilter(originalImage, kernel);
                    case '高斯滤波'
                        kernel = fspecial('gaussian', [5 5], 1);
                        processedImage = imfilter(originalImage, kernel);
                    case '中值滤波'
                        grayImage = rgb2gray(originalImage);
                        processedImage = medfilt2(grayImage, [3 3]);
                end

            case '直方图均衡化'
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end
                processedImage = customHistogramEqualization(grayImage);


            case '直方图匹配'
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end
                refImage = imread('bird.jpg'); % 示例参考图像路径
                if size(refImage, 3) == 3
                    refGrayImage = rgb2gray(refImage);
                else
                    refGrayImage = refImage;
                end
                processedImage = imhistmatch(grayImage, refGrayImage);
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

                % 提取 HOG 特征
                [hogFeatures, visualization] = extractHOGFeatures(grayImage);

                % 更新 processedImage 为特征提取后的图像
                processedImage = visualization; % 或者是通过 hogFeatures 可视化图像数据

                % 在处理后图像面板上显示更新后的 processedImage
                axes(axProcessed); % 将 axProcessed 设为当前坐标轴
                cla(axProcessed); % 清空当前坐标轴

                % 如果要显示 HOG 特征图像，可以使用 visualization 中的图形显示方法
                visualization.plot(); % 绘制 HOG 特征
                title(axProcessed, 'HOG 特征图像');

                % 更新 processedImage 用于后续的处理
                imshow(processedImage, 'Parent', axProcessed);
                disp('HOG 特征提取完成');

            case '目标提取'
                % 检查图像是否已加载
                if isempty(originalImage)
                    uialert(fig, '请先加载图像！', '错误');
                    return;
                end

                % 将图像转换为灰度图
                if size(originalImage, 3) == 3
                    grayImage = rgb2gray(originalImage);
                else
                    grayImage = originalImage;
                end

                % 先进行边缘检测（使用默认算子或菜单选择的算子）
                edgeImage = edge(grayImage, 'canny');

                % 调用目标提取函数
                targetImage = extractTargetFromEdges(edgeImage);

                % 保存并显示结果
                processedImage = targetImage;  % 更新全局变量
                imshow(processedImage, 'Parent', axProcessed);
                title(axProcessed, '目标提取结果');



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
        datasetPath = 'C:\Users\lili\Documents\WeChat Files\wxid_7qbo2zquoibf22\FileStorage\File\2024-12\CUB_200_2011\CUB_200_2011\images';
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
        % 如果是彩色图像，转换为灰度图像
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage; % 已经是灰度图像
    end

    % 使用 Robert 算子进行边缘检测
    edges_robert = edge(grayImage, 'roberts');
end



% Prewitt算子进行边缘检测
function edges_prewitt = applyPrewittEdgeDetection(inputImage)
     % 检查输入图像是否为灰度图像
    if size(inputImage, 3) == 3
        % 如果是彩色图像，转换为灰度图像
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage; % 已经是灰度图像
    end
    % 使用Prewitt算子进行边缘检测
    edges_prewitt = edge(grayImage, 'Prewitt');
end

% Sobel算子进行边缘检测
function edges_sobel = applySobelEdgeDetection(inputImage)
     % 检查输入图像是否为灰度图像
    if size(inputImage, 3) == 3
        % 如果是彩色图像，转换为灰度图像
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage; % 已经是灰度图像
    end
    % 使用Sobel算子进行边缘检测
    edges_sobel = edge(grayImage, 'Sobel');
end

%拉普拉斯算子进行边缘检测
function edges_laplacian = applyLaplacianEdgeDetection(inputImage)
     % 检查输入图像是否为灰度图像
    if size(inputImage, 3) == 3
        % 如果是彩色图像，转换为灰度图像
        grayImage = rgb2gray(inputImage);
    else
        grayImage = inputImage; % 已经是灰度图像
    end

    % 使用拉普拉斯算子进行边缘检测
    edges_laplacian = edge(grayImage, 'log'); % log 表示使用拉普拉斯高斯算子
end

%目标提取
function targetImage = extractTargetFromEdges(edgeImage)
    % 基于边缘图像提取目标
    % 使用形态学操作进行目标提取
    se = strel('disk', 5);  % 创建结构元素
    dilatedImage = imdilate(edgeImage, se);  % 膨胀操作
    
    % 填充连通区域（如果需要）
    targetImage = imfill(dilatedImage, 'holes');
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








