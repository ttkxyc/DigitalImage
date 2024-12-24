function ImageProcessingGUI()
    % 创建主界面
    fig = uifigure('Name', '图像处理系统', 'Position', [100, 100, 1000, 700]);

    % 添加网格布局
    grid = uigridlayout(fig, [3, 4]);
    grid.ColumnWidth = {'1x', '2x', '2x', '2x'};
    grid.RowHeight = {50, '3x', '3x'};

    % 文件加载按钮
    btnLoad = uibutton(grid, 'Text', '加载图像', 'ButtonPushedFcn', @loadImage);
    btnLoad.Layout.Row = 1;
    btnLoad.Layout.Column = 1;

    % 功能菜单
    ddFunction = uidropdown(grid, ...
        'Items', {'直方图均衡化', '灰度增强(线性)', '灰度增强(对数变换)', '灰度增强(指数变换)', ...
        '图像旋转', '添加噪声', '滤波处理', '边缘检测', ...
        '特征提取(LBP)', '特征提取(HOG)', '分类(深度学习)'});
    ddFunction.Layout.Row = 1;
    ddFunction.Layout.Column = 2;

    % 执行操作按钮
    btnProcess = uibutton(grid, 'Text', '执行操作', 'ButtonPushedFcn', @processImage);
    btnProcess.Layout.Row = 1;
    btnProcess.Layout.Column = 3;

    % 边缘检测算子选择菜单
    ddEdgeOperator = uidropdown(grid, ...
        'Items', {'Robert 算子', 'Prewitt 算子', 'Sobel 算子', '拉普拉斯算子'}, ...
        'Value', 'Sobel 算子', ...
        'Editable', 'off');
    ddEdgeOperator.Layout.Row = 1;
    ddEdgeOperator.Layout.Column = 4;

    % 原始图像面板
    panelOriginal = uipanel(grid, 'Title', '原始图像');
    panelOriginal.Layout.Row = 2;
    panelOriginal.Layout.Column = 2;
    axOriginal = uiaxes(panelOriginal);

    % 处理后图像面板
    panelProcessed = uipanel(grid, 'Title', '处理后图像');
    panelProcessed.Layout.Row = 2;
    panelProcessed.Layout.Column = 3;
    axProcessed = uiaxes(panelProcessed);

    % 加载图像函数
    function loadImage(~, ~)
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', '图像文件'});
        if isequal(file, 0)
            return; % 用户取消
        end
        imgPath = fullfile(path, file);
        originalImage = imread(imgPath);
        imshow(originalImage, 'Parent', axOriginal);
    end

    % 执行图像处理的回调函数
    function processImage(~, ~)
        if isempty(originalImage)
            uialert(fig, '请先加载图像！', '错误');
            return;
        end

        selectedFunction = ddFunction.Value;
        switch selectedFunction
            case '直方图均衡化'
                processedImage = equalize_histogram(originalImage);
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
            case '图像旋转'
                angle = 45; % 示例旋转角度
                processedImage = imrotate(originalImage, angle);
            case '添加噪声'
                processedImage = imnoise(originalImage, 'gaussian', 0, 0.01);
            case '滤波处理'
                h = fspecial('average', [5, 5]);
                processedImage = imfilter(originalImage, h);
            case '边缘检测'
                processedImage = detectEdges(originalImage);
            case '特征提取(LBP)'
                grayImage = rgb2gray(originalImage);
                processedImage = extractLBPFeatures(grayImage);
                uialert(fig, 'LBP特征提取完成！', '提示');
                return;
            case '特征提取(HOG)'
                grayImage = rgb2gray(originalImage);
                [features, visualization] = extractHOGFeatures(grayImage);
                imshow(grayImage, 'Parent', axProcessed);
                hold(axProcessed, 'on');
                plot(axProcessed, visualization, 'Color', 'green');
                hold(axProcessed, 'off');
                uialert(fig, 'HOG特征提取完成！', '提示');
                return;
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



% 非线性对比度增强：指数变换
function expEnhancedImage = exponentialContrastEnhancement(grayImage, c)
    % 指数变换增强对比度
    expEnhancedImage = c * (exp(double(grayImage)) - 1);
    
    % 确保输出值在有效的图像范围内 [0, 255]
    expEnhancedImage = uint8(min(max(expEnhancedImage, 0), 255));
end


% Robert算子进行边缘检测
function edges_robert = applyRobertEdgeDetection(grayImage)
    % 使用Robert算子进行边缘检测
    edges_robert = edge(grayImage, 'Robert');
end

% Prewitt算子进行边缘检测
function edges_prewitt = applyPrewittEdgeDetection(grayImage)
    % 使用Prewitt算子进行边缘检测
    edges_prewitt = edge(grayImage, 'Prewitt');
end

% Sobel算子进行边缘检测
function edges_sobel = applySobelEdgeDetection(grayImage)
    % 使用Sobel算子进行边缘检测
    edges_sobel = edge(grayImage, 'Sobel');
end

%拉普拉斯算子进行边缘检测
function edges_laplacian = applyLaplacianEdgeDetection(grayImage)
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

%LBP特征提取
function lbpFeatures = extractLBPFeatures(grayImage, radius, numNeighbors)
    % 使用MATLAB内置函数提取LBP特征
    % grayImage 是输入的灰度图像
    % radius 和 numNeighbors 是LBP的参数
    lbpImage = extractLBPFeatures(grayImage, 'Radius', radius, 'NumNeighbors', numNeighbors);
    
    % 返回LBP特征
    lbpFeatures = lbpImage;
end

%HOG特征提取
function hogFeatures = extractHOGFeatures(grayImage)
    % 提取HOG特征
    [hogFeatures, visualization] = extractHOGFeatures(grayImage, 'CellSize', [8 8]);
    
    % 返回HOG特征
    hogFeatures = hogFeatures;
end







