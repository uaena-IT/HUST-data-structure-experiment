#include "utils.h"

// 随机生成 K 个种子点，确保种子点分布较均匀
std::vector<cv::Point> generateSeedPoints(cv::Size size, int K) {
    std::vector<cv::Point> seeds;
    std::mt19937 rng((unsigned)time(nullptr));

    // 计算最小距离
    double minDistance = std::sqrt((size.width * size.height) / static_cast<double>(K));
    double minDistanceSquared = minDistance * minDistance; // 预计算平方距离

    // 随机生成初始点集
    std::uniform_real_distribution<double> x_dist(0, size.width);
    std::uniform_real_distribution<double> y_dist(0, size.height);

    // 随机选择第一个种子点
    seeds.push_back(cv::Point(x_dist(rng), y_dist(rng)));

    // 贪心策略：逐步选择距离当前种子点集合最远的点
    while (seeds.size() < K) {
        cv::Point bestCandidate;
        double maxMinDistance = -1.0;

        // 随机生成候选点并评估
        for (int attempt = 0; attempt < 100; ++attempt) { // 每轮尝试 100 个候选点
            cv::Point candidate(x_dist(rng), y_dist(rng));
            double minDistToSeeds = std::numeric_limits<double>::max();

            // 计算候选点到现有种子点的最小距离
            for (const auto& seed : seeds) {
                double distSquared = (candidate.x - seed.x) * (candidate.x - seed.x) +
                    (candidate.y - seed.y) * (candidate.y - seed.y);
                minDistToSeeds = std::min(minDistToSeeds, distSquared);
            }

            // 如果候选点的最小距离大于当前最大最小距离，则更新
            if (minDistToSeeds > maxMinDistance && minDistToSeeds >= minDistanceSquared) {
                maxMinDistance = minDistToSeeds;
                bestCandidate = candidate;
            }
        }

        // 如果找到合适的候选点，则加入种子点集合
        if (maxMinDistance >= minDistanceSquared) {
            seeds.push_back(bestCandidate);
        }
        else {
           // std::cout << "⚠️ 无法找到满足条件的候选点，放宽距离约束。" << std::endl;
            minDistance *= 0.95; // 动态放宽最小距离
            minDistanceSquared = minDistance * minDistance;
        }
    }

    // 输出调试信息
    //std::cout << "🔹 生成种子点完成，种子数：" << seeds.size() << std::endl;
    //for (size_t i = 0; i < seeds.size(); ++i) {
    //    std::cout << "Seed " << i + 1 << ": (" << seeds[i].x << ", " << seeds[i].y << ")" << std::endl;
    //}

    return seeds;
}



bool isPlanarGraph(const std::map<int, std::set<int>>& adjacency) {
    int V = adjacency.size(); // 顶点数
    int E = 0;               // 边数
    for (const auto& [node, neighbors] : adjacency) {
        E += neighbors.size();
    }
    E /= 2; // 每条边被计算了两次

    // 估算面数 F（假设为连通图）
    int F = 2 - V + E;

    // 检查 Euler 定理是否成立
    return (V - E + F == 2);
}


// 根据种子点创建 markers 图（CV_32S）
cv::Mat computeMarkers(cv::Size size, const std::vector<cv::Point>& seeds, const cv::Mat& src) {
    // 确保输入图像为 8 位 3 通道 (BGR)
    cv::Mat src_8uc3;
    if (src.type() != CV_8UC3) {
        src.convertTo(src_8uc3, CV_8UC3);
    }
    else {
        src_8uc3 = src.clone();
    }



    cv::Mat markers;
    std::map<int, std::set<int>> adjacency;

    while (true) {
        // 创建 markers 矩阵
        markers = cv::Mat::zeros(size, CV_32S);

        // 动态调整种子点半径
        int radius = std::max(3, static_cast<int>(std::sqrt((size.width * size.height) / (float)seeds.size()) * 0.3));
        std::cout << "自动计算种子半径：" << radius << std::endl;

        // 绘制种子点
        for (int i = 0; i < seeds.size(); ++i) {
            cv::circle(markers, seeds[i], radius, cv::Scalar(i + 1), -1);
        }

        // 转灰度图
        cv::Mat gray;
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray); // 增强对比度

        // 计算梯度图
        cv::Mat gradX, gradY, grad;
        cv::Sobel(gray, gradX, CV_16S, 1, 0, 3); // 水平方向梯度
        cv::Sobel(gray, gradY, CV_16S, 0, 1, 3); // 垂直方向梯度
        cv::convertScaleAbs(gradX, gradX);
        cv::convertScaleAbs(gradY, gradY);
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, grad); // 合并梯度
        cv::subtract(255, grad, grad); // 反转梯度值


        // 应用分水岭算法
        cv::watershed(src, markers);
        // 在分水岭算法后添加修复代码
        for (int y = 0; y < markers.rows; ++y) {
            for (int x = 0; x < markers.cols; ++x) {
                int& label = markers.at<int>(y, x);
                if (label <= 0) {
                    std::map<int, int> labelCount; // 统计邻域标签出现次数
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            int ny = y + dy, nx = x + dx;
                            if (ny >= 0 && ny < markers.rows && nx >= 0 && nx < markers.cols) {
                                int neighborLabel = markers.at<int>(ny, nx);
                                if (neighborLabel > 0) {
                                    labelCount[neighborLabel]++;
                                }
                            }
                        }
                    }
                    if (!labelCount.empty()) {
                        // 选择出现次数最多的标签
                        label = std::max_element(labelCount.begin(), labelCount.end(),
                            [](const auto& a, const auto& b) {
                                return a.second < b.second;
                            })->first;
                    }
                }
			}
		}

        // 查找最大标签
        int maxLabel = *std::max_element(markers.begin<int>(), markers.end<int>());
        int boundaryLabel = maxLabel + 1;

        // 构建邻接图
        adjacency.clear();
        for (int y = 0; y < markers.rows; ++y) {
            for (int x = 0; x < markers.cols; ++x) {




                if (markers.at<int>(y, x) <= 0) {
                    // 查找邻近像素的标签
                    std::set<int> neighborLabels;
                    if (x > 0) neighborLabels.insert(markers.at<int>(y, x - 1));
                    if (x < markers.cols - 1) neighborLabels.insert(markers.at<int>(y, x + 1));
                    if (y > 0) neighborLabels.insert(markers.at<int>(y - 1, x));
                    if (y < markers.rows - 1) neighborLabels.insert(markers.at<int>(y + 1));

                    // 使用邻近标签填充
                    if (!neighborLabels.empty()) {
                        markers.at<int>(y, x) = *neighborLabels.begin();
                    }
                }


                int label = markers.at<int>(y, x);
                if (label > 0) {
                    // 检查 4 邻域
                    if (x > 0 && markers.at<int>(y, x - 1) > 0 && markers.at<int>(y, x - 1) != label) {
                        adjacency[label].insert(markers.at<int>(y, x - 1));
                        adjacency[markers.at<int>(y, x - 1)].insert(label);
                    }
                    if (x < markers.cols - 1 && markers.at<int>(y, x + 1) > 0 && markers.at<int>(y, x + 1) != label) {
                        adjacency[label].insert(markers.at<int>(y, x + 1));
                        adjacency[markers.at<int>(y, x + 1)].insert(label);
                    }
                    if (y > 0 && markers.at<int>(y - 1, x) > 0 && markers.at<int>(y - 1, x) != label) {
                        adjacency[label].insert(markers.at<int>(y - 1, x));
                        adjacency[markers.at<int>(y - 1, x)].insert(label);
                    }
                    if (y < markers.rows - 1 && markers.at<int>(y + 1, x) > 0 && markers.at<int>(y + 1, x) != label) {
                        adjacency[label].insert(markers.at<int>(y + 1, x));
                        adjacency[markers.at<int>(y + 1, x)].insert(label);
                    }


                }




            }

        }

        // 检测是否为平面图
        if (isPlanarGraph(adjacency)) {
            //std::cout << "✅ 生成的图是平面图。" << std::endl;
            break;
        }
        else {
            std::cout << " 生成的图不是平面图，重新生成种子点。" << std::endl;
            // 重新生成种子点（可以调用 generateSeedPoints 或其他方法）
            // 注意：需要传入 size 和种子点数量 K
            // seeds = generateSeedPoints(size, seeds.size());
        }
    }

    //std::cout << " markers 完成，区域数：" << seeds.size() << "。" << std::endl;
    return markers;
}


// 应用分水岭算法，并返回彩色叠加图
cv::Mat applyWatershedWithColor1(const cv::Mat& src, cv::Mat& markers) {
    // 应用分水岭算法
    cv::watershed(src, markers);

    // 获取所有唯一的标签
    std::set<int> uniqueLabels;
    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            uniqueLabels.insert(markers.at<int>(y, x));
        }
    }

    // 为每个标签生成颜色映射
    std::map<int, cv::Vec3b> colorMap;
    cv::RNG rng(12345);
    for (int label : uniqueLabels) {
        if (label == -1) {
            colorMap[label] = cv::Vec3b(0, 0, 0); // 黑色边界
        }
        else {
            colorMap[label] = cv::Vec3b(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
        }
    }

    // 创建结果图像
    cv::Mat result = cv::Mat::zeros(markers.size(), CV_8UC3);

    // 使用矩阵操作填充颜色
    markers.forEach<int>([&](int& label, const int* position) {
        if (label == -1) {
            // 查找邻近像素的标签
            std::set<int> neighborLabels;
            int y = position[0], x = position[1];
            if (x > 0) neighborLabels.insert(markers.at<int>(y, x - 1));
            if (x < markers.cols - 1) neighborLabels.insert(markers.at<int>(y, x + 1));
            if (y > 0) neighborLabels.insert(markers.at<int>(y - 1, x));
            if (y < markers.rows - 1) neighborLabels.insert(markers.at<int>(y + 1, x));

            // 使用邻近标签的颜色
            if (!neighborLabels.empty()) {
                label = *neighborLabels.begin(); // 使用第一个邻近标签
            }
        }
        result.at<cv::Vec3b>(position[0], position[1]) = colorMap[label];
        });

    // 将结果与原图半透明融合
    cv::Mat blended;
    cv::addWeighted(src, 0.5, result, 0.5, 0, blended);

    std::cout << "✅ 分水岭区域图已生成并与原图半透明融合。" << std::endl;
    return blended;
}

cv::Mat applyWatershedWithColor(const cv::Mat& src, cv::Mat& markers) {
    // 应用分水岭算法
    cv::watershed(src, markers);

    // 获取所有唯一的标签
    std::set<int> uniqueLabels;
    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            uniqueLabels.insert(markers.at<int>(y, x));
        }
    }

    // 为每个标签生成颜色映射
    std::map<int, cv::Vec3b> colorMap;
    cv::RNG rng(12345);
    for (int label : uniqueLabels) {
        if (label == -1) {
            colorMap[label] = cv::Vec3b(0, 0, 0); // 黑色边界
        }
        else {
            colorMap[label] = cv::Vec3b(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
        }
    }

    // 创建结果图像
    cv::Mat result = cv::Mat::zeros(markers.size(), CV_8UC3);

    // 遍历每个像素，填充颜色
    for (int y = 0; y < markers.rows; ++y) {
        for (int x = 0; x < markers.cols; ++x) {
            int label = markers.at<int>(y, x);
            if (label == -1) {
                std::set<int> neighborLabels;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (nx >= 0 && nx < markers.cols && ny >= 0 && ny < markers.rows) {
                            int neighbor = markers.at<int>(ny, nx);
                            if (neighbor > 0) {
                                neighborLabels.insert(neighbor);
                            }
                        }
                    }
                }
                if (!neighborLabels.empty()) {
                    label = *neighborLabels.begin();
                }
            }

            else {
                result.at<cv::Vec3b>(y, x) = colorMap[label];
            }
        }
    }

    // 将结果与原图半透明融合
    cv::Mat blended;
    cv::addWeighted(src, 0.5, result, 0.5, 0, blended);

    //std::cout << "✅ 分水岭区域图已生成并与原图半透明融合。" << std::endl;
    return blended;
}


// 可视化种子点叠加原图（任务1展示用）
cv::Mat visualizeSeedOverlay(const cv::Mat& image, const std::vector<cv::Point>& seeds) {
    cv::Mat vis;
    image.copyTo(vis);
    for (size_t i = 0; i < seeds.size(); ++i) {
        cv::circle(vis, seeds[i], 4, cv::Scalar(255, 255, 255), -1);
        cv::putText(vis, std::to_string(i + 1), seeds[i] + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
    }
    return vis;
}


