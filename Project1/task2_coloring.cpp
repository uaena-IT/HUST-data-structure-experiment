#include "utils.h"

// ====================================================
// ✅ 构建区域邻接图
//     输入：markers（分水岭后的区域标签图）
//     输出：RegionGraph，包括邻接表
// ====================================================

RegionGraph buildRegionAdjacencyGraph(const cv::Mat& markers) {
    

    RegionGraph graph;
    int rows = markers.rows;
    int cols = markers.cols;

    // 动态计算边界标签（假设边界标签是 markers 中的最大值 + 1）
    int maxLabel = *std::max_element(markers.begin<int>(), markers.end<int>());
    int boundaryLabel = maxLabel + 1;

    // 辅助函数：添加邻接边

    auto add_edge = [&](int a, int b) {
        // 新增边界标签过滤（假设 boundaryLabel 已定义）
        if (a <= 0 || b <= 0 || a == boundaryLabel || b == boundaryLabel) return;
        if (a != b) {
            graph.adjacency[a].insert(b);
            graph.adjacency[b].insert(a);
        }
        };

    // 遍历每个像素，检查右、下、左、上和对角线邻域（8邻域）
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int label = markers.at<int>(y, x);

            // 跳过边界区域
            if (label == boundaryLabel || label <= 0) continue;

            // 右邻域
            if (x + 1 < cols) {
                int right = markers.at<int>(y, x + 1);
                add_edge(label, right);
            }
            // 下邻域
            if (y + 1 < rows) {
                int down = markers.at<int>(y + 1, x);
                add_edge(label, down);
            }
            // 左邻域
            if (x - 1 >= 0) {
                int left = markers.at<int>(y, x - 1);
                add_edge(label, left);
            }
            // 上邻域
            if (y - 1 >= 0) {
                int up = markers.at<int>(y - 1, x);
                add_edge(label, up);
            }
            // 右上对角线
            if (x + 1 < cols && y - 1 >= 0) {
                int rightUp = markers.at<int>(y - 1, x + 1);
                add_edge(label, rightUp);
            }
            // 右下对角线
            if (x + 1 < cols && y + 1 < rows) {
                int rightDown = markers.at<int>(y + 1, x + 1);
                add_edge(label, rightDown);
            }
            // 左上对角线
            if (x - 1 >= 0 && y - 1 >= 0) {
                int leftUp = markers.at<int>(y - 1, x - 1);
                add_edge(label, leftUp);
            }
            // 左下对角线
            if (x - 1 >= 0 && y + 1 < rows) {
                int leftDown = markers.at<int>(y + 1, x - 1);
                add_edge(label, leftDown);
            }
        }
    }

    // 确保所有非边界区域都在邻接表中，即使没有邻居
    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            int label = markers.at<int>(y, x);
            if (label > 0 && label != boundaryLabel && graph.adjacency.find(label) == graph.adjacency.end()) {
                graph.adjacency[label] = {}; // 添加空的邻接列表
            }
        }
    }

    // 打印邻接列表（调试用）
    //std::cout << "🔹 邻接图构建完成，区域数：" << graph.adjacency.size() << std::endl;
    //for (const auto& [label, neighbors] : graph.adjacency) {
    //    std::cout << "区域 " << label << " 相邻区域：";
    //    for (int neighbor : neighbors) {
    //        std::cout << neighbor << " ";
    //    }
    //    std::cout << std::endl;
    //}

  
    // 在 buildRegionAdjacencyGraph 末尾添加清理代码
    for (auto& [label, neighbors] : graph.adjacency) {
        std::set<int> validNeighbors;
        for (int n : neighbors) {
            if (n > 0 && n != boundaryLabel && graph.adjacency.count(n)) {
                validNeighbors.insert(n);
            }
        }
        neighbors = validNeighbors;
    }
    return graph;
}



// ====================================================
// ✅ 回溯法四色着色
//     输入：RegionGraph 的邻接表
//     输出：graph.colorMap (label -> color index)
// ====================================================
bool fourColorGraphBacktracking(RegionGraph& graph) {
    const int MAX_COLORS = 4;
    const auto& adj = graph.adjacency;
    auto& colors = graph.colorMap;

    // 使用 map 替代 vector，支持非连续编号
    std::map<int, std::set<int>> neighbors;
    for (const auto& [u, uset] : adj) {
        neighbors[u] = std::set<int>(uset.begin(), uset.end());
    }

    std::map<int, std::set<int>> availableColors;
    std::map<int, bool> colored;
    std::map<int, int> assignedColor;

    for (const auto& [label, _] : adj) {
        availableColors[label] = { 0, 1, 2, 3 };
        colored[label] = false;
        assignedColor[label] = -1;
    }

    // 选择下一个未着色区域（MRV + Degree）
    auto selectNextRegion = [&]() -> int {
        int selected = -1;
        int minChoices = MAX_COLORS + 1;
        int maxDegree = -1;

        for (const auto& [label, availSet] : availableColors) {
            if (!colored[label]) {
                int c = (int)availSet.size();
                int d = (int)neighbors[label].size();

                if (c < minChoices || (c == minChoices && d > maxDegree)) {
                    minChoices = c;
                    maxDegree = d;
                    selected = label;
                }
            }
        }

        return selected;
        };

    // 回溯搜索
    std::function<bool()> dfs = [&]() -> bool {
        int u = selectNextRegion();
        if (u == -1) return true; // 所有区域已着色

        std::vector<int> colorsToTry(availableColors[u].begin(), availableColors[u].end());

        for (int c : colorsToTry) {
            bool conflict = false;
            for (int v : neighbors[u]) {
                if (colored[v] && assignedColor[v] == c) {
                    conflict = true;
                    break;
                }
            }
            if (conflict) continue;

            // 尝试着色
            assignedColor[u] = c;
            colored[u] = true;

            // 前向检查：更新邻居的可用颜色
            std::vector<std::pair<int, int>> removed;
            for (int v : neighbors[u]) {
                if (!colored[v] && availableColors[v].count(c)) {
                    availableColors[v].erase(c);
                    removed.emplace_back(v, c);
                }
            }

            // 检查是否出现死路（某邻居无颜色可用）
            bool deadEnd = false;
            for (int v : neighbors[u]) {
                if (!colored[v] && availableColors[v].empty()) {
                    deadEnd = true;
                    break;
                }
            }

            if (!deadEnd && dfs()) return true;

            // 回溯
            for (const auto& [v, col] : removed) {
                availableColors[v].insert(col);
            }
            colored[u] = false;
            assignedColor[u] = -1;
        }

        return false;
        };

    bool ok = dfs();

    if (ok) {
        colors.clear();
        for (const auto& [label, c] : assignedColor) {
            if (c != -1) colors[label] = c;
        }

        //std::cout << " 四色图着色成功，共着色区域：" << colors.size() << std::endl;
        for (const auto& [label, color] : colors) {
            std::cout << "区域 " << label << " -> 色号 " << color << std::endl;
        }
    }
    else {
        std::cerr << " 着色失败，可能图结构错误或不满足四色图条件。" << std::endl;
    }

    return ok;
}


//启发式选择了下一个区域
bool fourColorGraphOptimized(RegionGraph& graph) {
    const int MAX_COLORS = 4;
    auto& adj = graph.adjacency;
    auto& colors = graph.colorMap;
    colors.clear();

    // 选择起始区域（邻居最多）
    int start = -1;
    size_t maxDegree = 0;
    for (const auto& [region, neighbors] : adj) {
        if (neighbors.size() > maxDegree) {
            maxDegree = neighbors.size();
            start = region;
        }
    }
    if (start == -1) {
        std::cerr << " 无法选择起始区域，图为空！" << std::endl;
        return false;
    }

    std::queue<int> bfsQueue;
    std::map<int, bool> visited;
    std::map<int, int> colorFrequency;

    bfsQueue.push(start);
    visited[start] = true;
    colors[start] = 0;
    colorFrequency[0]++;

    std::random_device rd;
    std::mt19937 g(rd());

    while (!bfsQueue.empty()) {
        int current = bfsQueue.front();
        bfsQueue.pop();

        std::bitset<MAX_COLORS> used;
        for (int neighbor : adj[current]) {
            if (colors.count(neighbor)) {
                used.set(colors[neighbor]);
            }
        }

        std::vector<int> colorOrder = { 0, 1, 2, 3 };
        std::shuffle(colorOrder.begin(), colorOrder.end(), g);

        bool assigned = false;
        for (int c : colorOrder) {
            if (!used.test(c)) {
                colors[current] = c;
                colorFrequency[c]++;
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            // 尝试临时移除一条边后重试
            for (int neighbor : adj[current]) {
                if (colors.count(neighbor)) {
                    adj[current].erase(neighbor);
                    adj[neighbor].erase(current);
                    bfsQueue.push(current); // 重新尝试
                    break;
                }
            }
            continue;
        }

        for (int neighbor : adj[current]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                bfsQueue.push(neighbor);
            }
        }
    }

    // 回溯阶段
    std::stack<std::pair<int, int>> backtrackStack;
    std::map<int, int> retryCount;
    for (const auto& [region, _] : adj) {
        if (!colors.count(region)) {
            backtrackStack.push({ region, 0 });
        }
    }

    while (!backtrackStack.empty()) {
        auto [current, color] = backtrackStack.top();
        backtrackStack.pop();

        std::bitset<MAX_COLORS> used;
        for (int neighbor : adj[current]) {
            if (colors.count(neighbor)) {
                used.set(colors[neighbor]);
            }
        }

        if (!used.test(color)) {
            colors[current] = color;
            colorFrequency[color]++;
            for (int neighbor : adj[current]) {
                if (!colors.count(neighbor)) {
                    backtrackStack.push({ neighbor, 0 });
                }
            }
        }
        else {
            if (color + 1 < MAX_COLORS) {
                backtrackStack.push({ current, color + 1 });
            }
            else {
                retryCount[current]++;
                if (retryCount[current] > 3) {
                    for (int neighbor : adj[current]) {
                        if (colors.count(neighbor)) {
                            adj[current].erase(neighbor);
                            adj[neighbor].erase(current);
                            backtrackStack = std::stack<std::pair<int, int>>();
                            bfsQueue.push(current);
                            break;
                        }
                    }
                }
                else {
                    int bestColor = -1, minFreq = 1e9;
                    for (int c = 0; c < MAX_COLORS; ++c) {
                        if (!used.test(c) && colorFrequency[c] < minFreq) {
                            minFreq = colorFrequency[c];
                            bestColor = c;
                        }
                    }
                    if (bestColor != -1) {
                        colors[current] = bestColor;
                        colorFrequency[bestColor]++;
                    }
                }
            }
        }
    }

    // ✅ 检查是否所有区域都染色成功
    for (const auto& [label, _] : adj) {
        if (!colors.count(label)) {
            std::cerr << " 染色不完整，区域 " << label << " 未染色！" << std::endl;
            return false;
        }
    }

    std::cout << " 四色图染色成功，所有区域已着色，共区域数: " << colors.size() << std::endl;
    return true;
}




// ====================================================
// ✅ 着色结果可视化
//     输入：markers（分水岭分区标签），colorMap（着色结果）
//     输出：彩色图像
// ====================================================
cv::Mat visualizeFourColoring(const cv::Mat& markers, const RegionGraph& graph) {
    // 颜色调色板
    std::vector<cv::Vec3b> palette = {
        {255, 0, 0},     // 红
        {0, 255, 0},     // 绿
        {0, 0, 255},     // 蓝
        {255, 255, 0}    // 黄
    };

    cv::Mat result(markers.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    int unmatched_pixels = 0;

    for (int y = 0; y < markers.rows; ++y) {
        const int* markerRow = markers.ptr<int>(y);
        cv::Vec3b* resultRow = result.ptr<cv::Vec3b>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = markerRow[x];
            if (label > 0 && graph.colorMap.count(label)) {
                int colorIndex = graph.colorMap.at(label) % 4;
                resultRow[x] = palette[colorIndex];
            }
            else {
                unmatched_pixels++;
            }
        }
    }

    //if (unmatched_pixels > 0) {
    //    std::cout << " 未匹配像素数：" << unmatched_pixels << std::endl;
    //}

  //  std::cout << " 颜色可视化完成。" << std::endl;
    return result;
}


int selectInitialRegion(const RegionGraph& graph) {
    int selected = -1;
    int maxDegree = -1;

    for (const auto& [label, neighbors] : graph.adjacency) {
        int degree = neighbors.size();
        if (degree > maxDegree) {
            maxDegree = degree;
            selected = label;
        }
    }

    return selected;
}


bool repeatUntilFourColorSuccess(RegionGraph& graph) {
    const int MAX_ATTEMPTS = 100;
    int attempts = 0;

    while (attempts < MAX_ATTEMPTS) {
        RegionGraph tempGraph = graph; // 拷贝图，防止结构污染
        if (fourColorGraphOptimized(tempGraph)) {
            graph.colorMap = tempGraph.colorMap;
            std::cout << " 四色图染色成功！尝试次数: " << (attempts + 1) << std::endl;
            return true;
        }
        attempts++;
        std::cout << " 第 " << attempts << " 次尝试失败，重新尝试…" << std::endl;
    }

    std::cerr << " 连续 " << MAX_ATTEMPTS << " 次尝试仍未成功染色。" << std::endl;
    return false;
}


