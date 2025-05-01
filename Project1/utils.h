#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <stack>
#include <bitset>
#include <algorithm>
#include <cmath>
#include <climits>
using namespace std;
using namespace cv;

// ========== 通用结构体 ==========
struct RegionGraph {
    std::map<int, std::set<int>> adjacency;  // 邻接表
    std::map<int, int> colorMap;             // 区域 label -> 颜色索引（0~3）
};

// ========== 任务1：分水岭 ==========
std::vector<cv::Point> generateSeedPoints(cv::Size size, int K);
cv::Mat computeMarkers(cv::Size size, const std::vector<cv::Point>& seeds, const cv::Mat& src);
cv::Mat applyWatershedWithColor(const cv::Mat& src, cv::Mat& markers);
cv::Mat visualizeSeedOverlay(const cv::Mat& image, const std::vector<cv::Point>& seeds);
bool isPlanarGraph(const std::map<int, std::set<int>>& adjacency);
// ========== 任务2：四色图着色 ==========
RegionGraph buildRegionAdjacencyGraph(const cv::Mat& markers);
bool fourColorGraphBacktracking(RegionGraph& graph);
cv::Mat visualizeFourColoring(const cv::Mat& markers, const RegionGraph& graph);
bool fourColorGraphOptimized(RegionGraph& graph);         
int selectInitialRegion(const RegionGraph& graph);
bool repeatUntilFourColorSuccess(RegionGraph& graph);
cv::Mat visualizeFourColoring(const cv::Mat& markers, const RegionGraph& graph);// ✅ 着色结果可视化


// ========== 任务3：哈夫曼编码 ==========

struct HuffmanNode {
    int weight;
    int label;         // 区域标签（仅叶子节点有效）
    HuffmanNode* left;
    HuffmanNode* right;
    HuffmanNode(int w, int l = -1) : weight(w), label(l), left(nullptr), right(nullptr) {}
};
struct AreaEntry {
    int label;
    int area;
};

extern std::vector<AreaEntry> sortedAreas;

cv::Mat visualizeHuffmanTree(HuffmanNode* root);
std::map<int, int> computeRegionAreas(const cv::Mat& markers);
void heapSortAndDisplay(std::map<int, int>& areaMap);
// utils.h 中修正声明
std::set<int> binarySearchInRange(const std::vector<AreaEntry>& sortedAreas, int low, int high);
void highlightRegions(
    cv::Mat& image,
    const cv::Mat& markers,
    const std::set<int>& targetLabels,
    const std::map<int, cv::Vec3b>& colorMap,
    const std::map<int, int>& areaMap,
    const std::map<int, cv::Point2f>& centerMap
);
HuffmanNode* buildHuffmanTree(const std::map<int, int>& areaMap);
void generateHuffmanCodes(HuffmanNode* root, std::string code, std::map<int, std::string>& codeMap);
void deleteHuffmanTree(HuffmanNode* root);
std::map<int, cv::Vec3b> generateColorMap(const std::set<int>& labels);
std::map<int, cv::Point2f> computeRegionCenters(
    const cv::Mat& markers,
    const std::map<int, int>& areaMap
);