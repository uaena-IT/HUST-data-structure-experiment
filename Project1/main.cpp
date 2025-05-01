#include "utils.h"
#include <chrono>

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    // -------- Step 0: 加载图像 --------
    cv::Mat src = cv::imread("wife.jpg");
    if (src.empty()) {
        std::cerr << " 无法读取图像文件 wife.jpg，请检查路径和文件是否存在。" << std::endl;
        return -1;
    }
    std::cout << " 图像加载成功，尺寸：" << src.cols << " x " << src.rows << "\n" << std::endl;

    // -------- Step 1: 分水岭分割 --------
    std::cout << "【任务1】分水岭分割 + 随机种子采样" << std::endl;
    std::cout << "请输入随机种子点个数 K（推荐100~1000）：";
    int K;
    std::cin >> K;
    if (K < 2 || K > 10000) {
        std::cerr << " 输入非法，K 应在 [2, 10000] 范围内。" << std::endl;
        return -1;
    }

    std::cout << "按下回车键开始任务1..." << std::endl;
    std::cin.ignore(); std::cin.get();
    auto t1_start = std::chrono::high_resolution_clock::now();

    std::vector<cv::Point> seeds = generateSeedPoints(src.size(), K);
    cv::Mat markers = computeMarkers(src.size(), seeds, src);
    cv::Mat seedOverlay = visualizeSeedOverlay(src, seeds);
    cv::Mat watershedView = applyWatershedWithColor(src, markers);

    auto t1_end = std::chrono::high_resolution_clock::now();
    std::cout << " 任务1完成，用时 "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t1_end - t1_start).count()
        << " ms\n" << std::endl;

    // 立即显示任务1结果并等待用户确认
    cv::imshow("任务1 - 原图与种子点叠加", seedOverlay);
    cv::imshow("任务1 - 分水岭区域图", watershedView);
    cv::waitKey(1); // 刷新窗口
    std::cout << "按回车键继续任务2..." << std::endl;
    std::cin.get();




    // -------- Step 2: 四色图着色 --------
    std::cout << "【任务2】四色图着色" << std::endl;
    auto t2_start = std::chrono::high_resolution_clock::now();

    RegionGraph graph = buildRegionAdjacencyGraph(markers);
    if (!repeatUntilFourColorSuccess(graph)) {
        std::cerr << " 四色着色失败，图结构可能异常。" << std::endl;
        return -1;
    }
    cv::Mat colorView = visualizeFourColoring(markers, graph);

    auto t2_end = std::chrono::high_resolution_clock::now();
    std::cout << " 任务2完成，用时 "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2_end - t2_start).count()
        << " ms\n" << std::endl;

    // 立即显示任务2结果并等待用户确认
    cv::imshow("任务2 - 四色着色图", colorView);
    cv::waitKey(1); // 刷新窗口
    std::cout << "按回车键继续任务3..." << std::endl;
    std::cin.get();

    // -------- Step 3: 面积排序 + 哈夫曼 --------
    std::cout << "【任务3】区域面积排序 + 哈夫曼编码" << std::endl;


    std::map<int, int> areaMap = computeRegionAreas(markers);
    if (areaMap.empty()) {
        std::cerr << " 区域面积计算失败，无法继续任务3。" << std::endl;
        return -1;
    }

    heapSortAndDisplay(areaMap);

    int low, high;
    std::cout << "请输入面积下限：";
    while (!(std::cin >> low) || low < 0) {
        std::cin.clear(); std::cin.ignore(INT_MAX, '\n');
        std::cout << " 无效输入，请输入非负整数：";
    }
    std::cout << "请输入面积上限：";
    while (!(std::cin >> high) || high < low) {
        std::cin.clear(); std::cin.ignore(INT_MAX, '\n');
        std::cout << " 无效输入，上限应 ≥ 下限：";
    }
    auto t3_start = std::chrono::high_resolution_clock::now();
    std::vector<AreaEntry> sortedAreas;
    for (const auto& [label, area] : areaMap)
        sortedAreas.push_back({ label, area });
    std::sort(sortedAreas.begin(), sortedAreas.end(),
        [](const AreaEntry& a, const AreaEntry& b) { return a.area < b.area; });

    std::set<int> targetLabels = binarySearchInRange(sortedAreas, low, high);
    std::cout << " 共找到 " << targetLabels.size() << " 个区域符合条件。\n" << std::endl;

    auto colorMap = generateColorMap(targetLabels);
    auto centerMap = computeRegionCenters(markers, areaMap);
    cv::Mat highlightedImage = src.clone();
    highlightRegions(highlightedImage, markers, targetLabels, colorMap, areaMap, centerMap);
    cv::imshow("任务3 - 高亮显示目标区域", highlightedImage);

    std::map<int, int> filteredAreaMap;
    for (const auto& entry : sortedAreas) {
        if (entry.area >= low && entry.area <= high)
            filteredAreaMap[entry.label] = entry.area;
    }
    HuffmanNode* huffmanTree = buildHuffmanTree(filteredAreaMap);
    if (!huffmanTree) {
        std::cerr << " 哈夫曼树构建失败！" << std::endl;
        return -1;
    }

    std::map<int, std::string> huffmanCodes;
    generateHuffmanCodes(huffmanTree, "", huffmanCodes);
    //std::cout << " 哈夫曼编码：" << std::endl;
    //for (const auto& [label, code] : huffmanCodes) {
    //    std::cout << "区域 " << label << " (面积=" << areaMap[label] << ") -> " << code << std::endl;
    //}

    cv::Mat huffmanView = visualizeHuffmanTree(huffmanTree);
    cv::imshow("任务3 - 哈夫曼树可视化", huffmanView);

    auto t3_end = std::chrono::high_resolution_clock::now();
    std::cout << "\n 任务3完成，用时 "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t3_end - t3_start).count()
        << " ms\n" << std::endl;

    // 立即显示任务3结果并等待用户确认
    cv::waitKey(1); // 刷新窗口
    std::cout << " 所有任务执行完毕！按任意键退出程序。" << std::endl;
    cv::waitKey(0);

    // -------- 释放资源 --------
    deleteHuffmanTree(huffmanTree);
    return 0;
}