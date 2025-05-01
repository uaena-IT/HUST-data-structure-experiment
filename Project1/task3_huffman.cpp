#include "utils.h"
std::vector<AreaEntry> sortedAreas;



// 统计各区域面积
std::map<int, int> computeRegionAreas(const cv::Mat& markers) {
    std::map<int, int> areaMap;
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = row[x];
            if (label > 0) { // 过滤无效标签（边界或未分配区域）
                areaMap[label]++;
            }
        }
    }
    return areaMap;
}




// 生成标签到随机颜色的映射
std::map<int, cv::Vec3b> generateColorMap(const std::set<int>& labels) {
    std::map<int, cv::Vec3b> colorMap;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(50, 255); // 避免颜色过暗

    for (int label : labels) {
        colorMap[label] = cv::Vec3b(
            dis(gen), // B通道
            dis(gen), // G通道
            dis(gen)  // R通道
        );
    }
    return colorMap;
}




// 计算每个区域的质心坐标
std::map<int, cv::Point2f> computeRegionCenters(
    const cv::Mat& markers,
    const std::map<int, int>& areaMap
) {
    std::map<int, cv::Point2f> centerMap;
    std::map<int, cv::Moments> momentsMap;

    // 计算每个区域的矩
    for (int y = 0; y < markers.rows; ++y) {
        const int* row = markers.ptr<int>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = row[x];
            if (label > 0 && areaMap.count(label)) {
                cv::Moments m = momentsMap[label];
                m.m00 += 1; // 累加像素数
                m.m10 += x; // 累加 x 坐标
                m.m01 += y; // 累加 y 坐标
                momentsMap[label] = m; // 更新 momentsMap
            }
        }
    }


    // 计算质心
    for (auto& [label, m] : momentsMap) {
        if (m.m00 != 0) {
            centerMap[label] = cv::Point2f(
                m.m10 / m.m00, // x坐标
                m.m01 / m.m00  // y坐标
            );
        }
    }
    return centerMap;
}



// 堆排序并输出最大/最小面积
void heapSortAndDisplay(std::map<int, int>& areaMap) {
    if (areaMap.empty()) {
        std::cerr << "⚠️ 区域面积映射为空，请检查输入数据！" << std::endl;
        return;
    }

    // 提取面积值到向量
    std::vector<int> areas;
    for (const auto& [label, area] : areaMap) {
        areas.push_back(area);
    }

    // 构建最大堆
    std::make_heap(areas.begin(), areas.end());

    // 输出最大值（堆顶）
    std::cout << "✅ 最大区域面积: " << areas.front() << std::endl;

    // 遍历找最小值
    int minArea = INT_MAX;
    for (const auto& [label, area] : areaMap) {
        if (area < minArea) {
            minArea = area;
        }
    }
    std::cout << "✅ 最小区域面积: " << minArea << std::endl;

    // 完整堆排序（可选）
    // std::sort_heap(areas.begin(), areas.end());
}



// 二分查找符合面积范围的区域标签集合
std::set<int> binarySearchInRange(const std::vector<AreaEntry>& sortedAreas, int low, int high) {
    std::set<int> targetLabels;
    if (sortedAreas.empty()) return targetLabels;

    auto lower = std::lower_bound(sortedAreas.begin(), sortedAreas.end(), low,
        [](const AreaEntry& a, int value) { return a.area < value; });

    auto upper = std::upper_bound(sortedAreas.begin(), sortedAreas.end(), high,
        [](int value, const AreaEntry& a) { return value < a.area; });



    for (auto it = lower; it != upper; ++it) {
        targetLabels.insert(it->label);
    }
    return targetLabels;
}



// 高亮显示目标区域
void highlightRegions1(cv::Mat& image, const cv::Mat& markers, const std::set<int>& targetLabels) {
    if (image.empty() || markers.empty()) {
        std::cerr << "⚠️ 输入图像或标记矩阵为空！" << std::endl;
        return;
    }

    // 定义高亮颜色（红色）
    const cv::Vec3b HIGHLIGHT_COLOR(0, 0, 255);

    // 遍历标记矩阵，高亮目标标签区域
    for (int y = 0; y < markers.rows; ++y) {
        const int* markersRow = markers.ptr<int>(y);
        cv::Vec3b* imageRow = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = markersRow[x];
            if (targetLabels.count(label)) {
                imageRow[x] = HIGHLIGHT_COLOR; // BGR格式
            }
        }
    }
}


void highlightRegions(
    cv::Mat& image,
    const cv::Mat& markers,
    const std::set<int>& targetLabels,
    const std::map<int, cv::Vec3b>& colorMap,
    const std::map<int, int>& areaMap,
    const std::map<int, cv::Point2f>& centerMap
) {
    // 高亮区域颜色
    for (int y = 0; y < markers.rows; ++y) {
        const int* markersRow = markers.ptr<int>(y);
        cv::Vec3b* imageRow = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < markers.cols; ++x) {
            int label = markersRow[x];
            if (targetLabels.count(label)) {
                imageRow[x] = colorMap.at(label);
            }
        }
    }

    // 标注面积值
    for (const auto& [label, center] : centerMap) {
        if (targetLabels.count(label)) {
            std::string text = std::to_string(areaMap.at(label));
            cv::putText(image, text, center,
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 0, 0), 2); // 黑色文字，粗体
        }
    }
}




// ================== 哈夫曼树构建 ==================
HuffmanNode* buildHuffmanTree(const std::map<int, int>& areaMap) {
    // 自定义优先队列比较函数（按权值升序）
    auto cmp = [](HuffmanNode* a, HuffmanNode* b) {
        return a->weight > b->weight;
        };
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, decltype(cmp)> minHeap(cmp);

    // 创建叶子节点（每个区域对应一个叶子）
    for (const auto& [label, area] : areaMap) {
        minHeap.push(new HuffmanNode(area, label)); // 保存区域标签和面积
    }

    // 合并节点直到只剩根节点
    while (minHeap.size() > 1) {
        // 取出权值最小的两个节点
        HuffmanNode* left = minHeap.top();
        minHeap.pop();
        HuffmanNode* right = minHeap.top();
        minHeap.pop();

        // 创建父节点（权值为子节点之和，标签无效）
        HuffmanNode* parent = new HuffmanNode(left->weight + right->weight);
        parent->left = left;
        parent->right = right;

        minHeap.push(parent);
    }

    return minHeap.empty() ? nullptr : minHeap.top();
}



// ================== 哈夫曼编码生成 ==================
void generateHuffmanCodes(HuffmanNode* root, std::string code, std::map<int, std::string>& codeMap) {
    if (!root) return;

    // 叶子节点：记录标签对应的编码
    if (!root->left && !root->right) {
        codeMap[root->label] = code; // 标签与编码关联
        return;
    }

    // 递归左子树（编码追加"0"）
    generateHuffmanCodes(root->left, code + "0", codeMap);
    // 递归右子树（编码追加"1"）
    generateHuffmanCodes(root->right, code + "1", codeMap);
}



// ================== 哈夫曼树可视化 ==================
cv::Mat visualizeHuffmanTree1(HuffmanNode* root) {
    const int NODE_RADIUS = 20;        // 节点圆的半径
    const int HORIZONTAL_SPACING = 60; // 水平间距（兄弟节点间）
    const int VERTICAL_SPACING = 80;   // 垂直间距（父子节点间）
    const cv::Scalar NODE_COLOR(255, 255, 255);  // 节点颜色（白色）
    const cv::Scalar LINE_COLOR(0, 200, 0);       // 连线颜色（绿色）
    const cv::Scalar TEXT_COLOR(0, 0, 0);         // 文本颜色（黑色）

    // ---------------------- 辅助结构：存储节点位置信息 ----------------------
    struct NodePosition {
        HuffmanNode* node;
        cv::Point center;
        int depth;
        NodePosition(HuffmanNode* n, cv::Point c, int d) : node(n), center(c), depth(d) {}
    };

    // ---------------------- 递归计算节点位置 ----------------------
    std::vector<NodePosition> positions;
    std::function<void(HuffmanNode*, cv::Point, int, int)> calculatePosition =
        [&](HuffmanNode* node, cv::Point parentPos, int depth, int horizontalOffset) {
        if (!node) return;

        // 计算当前节点位置（根节点居中，子节点按偏移量分布）
        cv::Point currentPos;
        if (depth == 0) {
            // 根节点位于画布顶部中央
            currentPos = cv::Point(horizontalOffset, NODE_RADIUS + 10);
        }
        else {
            currentPos = cv::Point(
                parentPos.x + horizontalOffset,
                parentPos.y + VERTICAL_SPACING
            );
        }
        positions.emplace_back(node, currentPos, depth);

        // 递归计算左右子节点位置（右子节点向右偏移，左子节点向左偏移）
        calculatePosition(node->left, currentPos, depth + 1, -HORIZONTAL_SPACING);
        calculatePosition(node->right, currentPos, depth + 1, HORIZONTAL_SPACING);
        };

    // 初始调用：从根节点开始计算位置
    calculatePosition(root, cv::Point(0, 0), 0, 0);

    // ---------------------- 动态计算画布大小 ----------------------
    int maxX = 0, minX = 0, maxDepth = 0;
    for (const auto& pos : positions) {
        maxX = std::max(maxX, pos.center.x);
        minX = std::min(minX, pos.center.x);
        maxDepth = std::max(maxDepth, pos.depth);
    }
    int imgWidth = (maxX - minX) + 4 * NODE_RADIUS;
    int imgHeight = (maxDepth + 1) * VERTICAL_SPACING + 2 * NODE_RADIUS;

    // 创建画布（白色背景）
    cv::Mat treeImage(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    // ---------------------- 绘制连线和节点 ----------------------
    for (const auto& pos : positions) {
        HuffmanNode* node = pos.node;
        cv::Point center(pos.center.x - minX + 2 * NODE_RADIUS, pos.center.y);

        // 绘制连线到子节点
        if (node->left) {
            cv::Point leftChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->left) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, leftChildCenter, LINE_COLOR, 2);
        }
        if (node->right) {
            cv::Point rightChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->right) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, rightChildCenter, LINE_COLOR, 2);
        }

        // 绘制节点圆
        cv::circle(treeImage, center, NODE_RADIUS, NODE_COLOR, -1);
        cv::circle(treeImage, center, NODE_RADIUS, LINE_COLOR, 2);

        // 添加文本（权值和标签）
        std::string text;
        if (node->left || node->right) {
            text = std::to_string(node->weight); // 内部节点显示权值
        }
        else {
            text = "L" + std::to_string(node->label) + "\n" + std::to_string(node->weight);
        }
        cv::putText(treeImage, text, cv::Point(center.x - 15, center.y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    }

    return treeImage;
}


cv::Mat visualizeHuffmanTree2(HuffmanNode* root) {
    const int NODE_RADIUS = 20;        // 节点圆的半径
    const int HORIZONTAL_SPACING = 100; // 增大水平间距
    const int VERTICAL_SPACING = 120;   // 增大垂直间距
    const cv::Scalar NODE_COLOR(255, 255, 255);  // 节点颜色（白色）
    const cv::Scalar LINE_COLOR(0, 200, 0);       // 连线颜色（绿色）
    const cv::Scalar TEXT_COLOR(0, 0, 0);         // 文本颜色（黑色）

    struct NodePosition {
        HuffmanNode* node;
        cv::Point center;
        int depth;
        NodePosition(HuffmanNode* n, cv::Point c, int d) : node(n), center(c), depth(d) {}
    };

    std::vector<NodePosition> positions;
    std::function<void(HuffmanNode*, cv::Point, int, int)> calculatePosition =
        [&](HuffmanNode* node, cv::Point parentPos, int depth, int horizontalOffset) {
        if (!node) return;

        cv::Point currentPos;
        if (depth == 0) {
            currentPos = cv::Point(0, NODE_RADIUS + 10);
        }
        else {
            currentPos = cv::Point(
                parentPos.x + horizontalOffset,
                parentPos.y + VERTICAL_SPACING
            );
        }
        positions.emplace_back(node, currentPos, depth);

        calculatePosition(node->left, currentPos, depth + 1, -HORIZONTAL_SPACING / (depth + 1));
        calculatePosition(node->right, currentPos, depth + 1, HORIZONTAL_SPACING / (depth + 1));
        };

    calculatePosition(root, cv::Point(0, 0), 0, 0);

    int maxX = 0, minX = 0, maxDepth = 0;
    for (const auto& pos : positions) {
        maxX = std::max(maxX, pos.center.x);
        minX = std::min(minX, pos.center.x);
        maxDepth = std::max(maxDepth, pos.depth);
    }
    int imgWidth = (maxX - minX) + 4 * NODE_RADIUS;
    int imgHeight = (maxDepth + 1) * VERTICAL_SPACING + 2 * NODE_RADIUS;

    cv::Mat treeImage(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const auto& pos : positions) {
        HuffmanNode* node = pos.node;
        cv::Point center(pos.center.x - minX + 2 * NODE_RADIUS, pos.center.y);

        if (node->left) {
            cv::Point leftChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->left) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, leftChildCenter, LINE_COLOR, 2);
        }
        if (node->right) {
            cv::Point rightChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->right) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, rightChildCenter, LINE_COLOR, 2);
        }

        cv::circle(treeImage, center, NODE_RADIUS, NODE_COLOR, -1);
        cv::circle(treeImage, center, NODE_RADIUS, LINE_COLOR, 2);

        std::string text;
        if (node->left || node->right) {
            text = std::to_string(node->weight);
        }
        else {
            text = "L" + std::to_string(node->label) + "\n" + std::to_string(node->weight);
        }
        cv::putText(treeImage, text, cv::Point(center.x - 15, center.y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
    }

    return treeImage;
}

cv::Mat visualizeHuffmanTree(HuffmanNode* root) {
    const int NODE_RADIUS = 20;        // 节点圆的半径
    const int HORIZONTAL_SPACING = 50; // 增大水平间距
    const int VERTICAL_SPACING = 50;   // 增大垂直间距
    const cv::Scalar NODE_COLOR(255, 255, 255);  // 节点颜色（白色）
    const cv::Scalar LINE_COLOR(0, 200, 0);       // 连线颜色（绿色）
    const cv::Scalar TEXT_COLOR(0, 0, 0);         // 文本颜色（黑色）

    struct NodePosition {
        HuffmanNode* node;
        cv::Point center;
        int depth;
        NodePosition(HuffmanNode* n, cv::Point c, int d) : node(n), center(c), depth(d) {}
    };
    
    // 计算子树宽度（递归计算每个节点的子树宽度）
    std::function<int(HuffmanNode*)> calculateSubtreeWidth = [&](HuffmanNode* node) -> int {
        if (!node) return 0;

        // 叶子节点宽度为 1
        if (!node->left && !node->right) return 1;

        // 子树宽度为左右子树宽度之和
        return calculateSubtreeWidth(node->left) + calculateSubtreeWidth(node->right);
        };


    std::vector<NodePosition> positions;
    std::function<void(HuffmanNode*, cv::Point, int)> calculatePosition =
        [&](HuffmanNode* node, cv::Point parentPos, int depth) {
        if (!node) return;

        // 计算当前节点位置
        cv::Point currentPos = cv::Point(
            parentPos.x,
            parentPos.y + VERTICAL_SPACING
        );
        positions.emplace_back(node, currentPos, depth);

        // 计算左右子树宽度
        int leftWidth = calculateSubtreeWidth(node->left);
        int rightWidth = calculateSubtreeWidth(node->right);

        // 动态调整水平偏移量
        int leftOffset = -(leftWidth * NODE_RADIUS * 2); // 左子树向左偏移
        int rightOffset = rightWidth * NODE_RADIUS * 2;  // 右子树向右偏移

        // 递归计算左右子节点位置
        calculatePosition(node->left, cv::Point(currentPos.x + leftOffset, currentPos.y), depth + 1);
        calculatePosition(node->right, cv::Point(currentPos.x + rightOffset, currentPos.y), depth + 1);
        };


     calculatePosition(root, cv::Point(500, NODE_RADIUS + 10), 0);


    int maxX = 0, minX = 0, maxDepth = 0;
    for (const auto& pos : positions) {
        maxX = std::max(maxX, pos.center.x);
        minX = std::min(minX, pos.center.x);
        maxDepth = std::max(maxDepth, pos.depth);
    }
    int imgWidth = (maxX - minX) + 10 * NODE_RADIUS;
    int imgHeight = (maxDepth + 1) * VERTICAL_SPACING + 4 * NODE_RADIUS;

    cv::Mat treeImage(imgHeight, imgWidth, CV_8UC3, cv::Scalar(255, 255, 255));

    for (const auto& pos : positions) {
        HuffmanNode* node = pos.node;
        cv::Point center(pos.center.x - minX + 2 * NODE_RADIUS, pos.center.y);

        if (node->left) {
            cv::Point leftChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->left) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, leftChildCenter, LINE_COLOR, 2);
        }
        if (node->right) {
            cv::Point rightChildCenter = [&]() {
                for (const auto& childPos : positions) {
                    if (childPos.node == node->right) {
                        return cv::Point(
                            childPos.center.x - minX + 2 * NODE_RADIUS,
                            childPos.center.y
                        );
                    }
                }
                return cv::Point(0, 0);
                }();
            cv::line(treeImage, center, rightChildCenter, LINE_COLOR, 2);
        }

        cv::circle(treeImage, center, NODE_RADIUS, NODE_COLOR, -1);
        cv::circle(treeImage, center, NODE_RADIUS, LINE_COLOR, 2);



        // 添加文本（权值和标签）
        std::string text;
        if (node->left || node->right) {
            text = std::to_string(node->weight); // 内部节点显示权值
        }
        else {
            // 叶子节点：格式为 "L{label}\n{weight}"
            text = "L" + std::to_string(node->label) + "\n" + std::to_string(node->weight);
        }

        // 拆分多行文本并逐行绘制
        std::vector<std::string> lines;
        size_t pos = text.find('\n');
        if (pos != std::string::npos) {
            lines.push_back(text.substr(0, pos));
            lines.push_back(text.substr(pos + 1));
        }
        else {
            lines.push_back(text);
        }

        // 计算总文本高度
        int totalHeight = 0;
        int baseline = 0;
        for (const auto& line : lines) {
            cv::Size textSize = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            totalHeight += textSize.height + 5; // 行间距
        }

        // 逐行绘制
        int currentY = center.y - totalHeight / 2;
        for (const auto& line : lines) {
            cv::Size textSize = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point textPos(
                center.x - textSize.width / 2,
                currentY + textSize.height
            );
            cv::putText(treeImage, line, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1);
            currentY += textSize.height + 5;
        }


    }

    return treeImage;
}






// ================== 释放哈夫曼树内存 ==================
void deleteHuffmanTree(HuffmanNode* root) {
    if (!root) return;
    deleteHuffmanTree(root->left);
    deleteHuffmanTree(root->right);
    delete root;
}