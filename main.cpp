#define OPEN_IMAGE_ERROR (-2)
#include <iostream>
#include <opencv2\opencv.hpp>
#include <random>

void fill_labels(cv::Mat *label, cv::Mat *label_u, cv::Mat *label_d, cv::Mat *label_l, cv::Mat *label_r,
                 cv::Mat *label_ul, cv::Mat *label_ur, cv::Mat *label_dl, cv::Mat *label_dr);
void release_labels(cv::Mat *label_u, cv::Mat *label_d, cv::Mat *label_l, cv::Mat *label_r,
                    cv::Mat *label_ul, cv::Mat *label_ur, cv::Mat *label_dl, cv::Mat *label_dr);
void print_mat(cv::Mat *image);
void show_image(cv:: Mat *label, int cluster_num);
void save_image(cv:: Mat *label, int cluster_num);

// https://habr.com/ru/articles/501850/
int main() {
    std::cout << "Hello, World!" << std::endl;

    cv::Mat color_image = cv::imread("../lenna.png");
    if (color_image.empty()) {
        std::cout << "Could not open or find the color_image" << std::endl;
        return OPEN_IMAGE_ERROR;
    }

    cv::Mat image;
    cv::cvtColor(color_image, image, cv::COLOR_BGR2GRAY);
    color_image.release();

    int width = image.cols;
    int height = image.rows;
    int image_size = width * height;
    int cluster_num = 3;
    int max_iter = 30;

    cv::Mat label(height, width, CV_8U);
    cv::RNG rng;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            label.at<uint8_t>(y, x) = rng.uniform(0, cluster_num);
        }
    }

    cv::Mat label_u, label_d, label_l, label_r, label_ul, label_ur, label_dl, label_dr;

    for (int k = 0; k < max_iter; ++k) {
        printf("Iter number: %d\n", k);
        fill_labels(&label, &label_u, &label_d, &label_l, &label_r, &label_ul, &label_ur, &label_dl, &label_dr);
        // объявление матрицы p_c и заполнение нулями
        std::vector<std::vector<double>> p_c(cluster_num, std::vector<double>(width * height, 0.0));
        // вычисление вероятностей для каждого класса сегментации
        for (int i = 0; i < cluster_num; i++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = y * width + x;
                    int contributors = 0;
                    if (x > 0 && y > 0) {
                        contributors++;
                        if (label.at<uchar>(y - 1, x - 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (y - 1 >= 0) {
                        contributors++;
                        if (label.at<uchar>(y - 1,x) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (y - 1 >= 0 && x + 1 < width) {
                        contributors++;
                        if (label.at<uchar>(y - 1, x + 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (x - 1 >= 0) {
                        contributors++;
                        if (label.at<uchar>(y, x - 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (x + 1 < width) {
                        contributors++;
                        if (label.at<uchar>(y, x + 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (x - 1 >= 0 && y + 1 < height) {
                        contributors++;
                        if (label.at<uchar>(y + 1, x - 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (y + 1 < height) {
                        contributors++;
                        if (label.at<uchar>(y + 1, x) == i) {
                            p_c[i][index]++;
                        }
                    }
                    if (x + 1 < width && y + 1 < height) {
                        contributors++;
                        if (label.at<uchar>(y + 1, x + 1) == i) {
                            p_c[i][index]++;
                        }
                    }
                    p_c[i][index] /= contributors;
                }
            }
        }
        // объявление массивов mu и sigma и заполнение нулями
        std::vector<double> mu(cluster_num, 0.0);
        std::vector<double> sigma(cluster_num, 0.0);

        // вычисление среднего и дисперсии для каждого класса сегментации
        for (int i = 0; i < cluster_num; i++) {
            // получение индексов пикселей в текущем классе сегментации
            std::vector<int> index;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (label.at<uchar>(y, x) == i) {
                        index.push_back(y * width + x);
                    }
                }
            }

            // вычисление среднего и дисперсии яркости пикселей в текущем классе сегментации
            std::vector<int> data_c(index.size());
            for (int j = 0; j < index.size(); j++) {
                data_c[j] = image.at<uchar>(index[j]/width, index[j]%width);
            }
            double sum = 0.0;
            for (int j : data_c) {
                sum += j;
            }
            double mean = sum / (double) data_c.size();
            double var = 0.0;
            for (int j : data_c) {
                var += pow(j - mean, 2);
            }
            var /= (double) (data_c.size() - 1);

            // сохранение результатов для текущего класса сегментации
            mu[i] = mean;
            sigma[i] = var;
        }

        std::vector<std::vector<double>> p_sc(cluster_num, std::vector<double>(image_size, 0.0));
        // вычисление вероятности для каждого пикселя и каждого класса сегментации
        for (int j = 0; j < cluster_num; j++) {
            // вычисление среднего значения яркости пикселей для текущего класса сегментации

            // вычисление плотности вероятности нормального распределения для каждого пикселя
            double coef = 1.0 / sqrt(2.0 * M_PI * sigma[j]);
            for (int i = 0; i < image_size; i++) {
                double diff = double(image.at<uchar>(i/width, i%width)) - mu[j];
                double prob = coef * exp(-diff*diff/(2.0*sigma[j]));
                p_sc[j][i] = prob;
            }
        }

        // нахождение метки класса с максимальной вероятностью для каждого пикселя
        for (int i = 0; i < image_size; i++) {
            double max_prob = 0.0;
            for (int j = 0; j < cluster_num; j++) {
                double prob = p_sc[j][i] * p_c[j][i];
                if (prob > max_prob) {
                    max_prob = prob;
                    label.at<uchar>(i/width, i%width) = j;
                }
            }
        }
        std::vector<std::vector<double>>().swap(p_c);
        std::vector<std::vector<double>>().swap(p_sc);
        std::vector<double>().swap(mu);
        std::vector<double>().swap(sigma);
        show_image(&label, cluster_num);
    }
    release_labels(&label_u, &label_d, &label_l, &label_r, &label_ul, &label_ur, &label_dl, &label_dr);
    save_image(&label, cluster_num);
    label.release();
    return 0;
}

void fill_labels(cv::Mat *label, cv::Mat *label_u, cv::Mat *label_d, cv::Mat *label_l, cv::Mat *label_r,
                 cv::Mat *label_ul, cv::Mat *label_ur, cv::Mat *label_dl, cv::Mat *label_dr) {
    cv::Mat kernel = (cv::Mat_<float>(3,3) << 0, 1, 0, 0, 0, 0, 0, 0, 0);
    cv::filter2D(*label, *label_u, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 0, 1, 0, 0, 0, 0, 0);
    cv::filter2D(*label, *label_l, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 0, 1, 0, 0, 0);
    cv::filter2D(*label, *label_r, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0, 1, 0);
    cv::filter2D(*label, *label_d, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 1, 0, 0, 0, 0, 0, 0, 0, 0);
    cv::filter2D(*label, *label_ul, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 1, 0, 0, 0, 0, 0, 0);
    cv::filter2D(*label, *label_ur, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 1, 0, 0);
    cv::filter2D(*label, *label_dl, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel = (cv::Mat_<float>(3,3) << 0, 0, 0, 0, 0, 0, 0, 0, 1);
    cv::filter2D(*label, *label_dr, label->depth(), kernel,
                 cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);
    kernel.release();
}

void release_labels(cv::Mat *label_u, cv::Mat *label_d, cv::Mat *label_l, cv::Mat *label_r,
                    cv::Mat *label_ul, cv::Mat *label_ur, cv::Mat *label_dl, cv::Mat *label_dr) {
    label_u->release();
    label_d->release();
    label_l->release();
    label_r->release();
    label_ur->release();
    label_ul->release();
    label_dl->release();
    label_dr->release();
}

void show_image(cv:: Mat *label, int cluster_num) {
    int step = 256 / cluster_num;

    cv::Mat dst = cv::Mat::zeros(label->size(), CV_8U);
    for (int y = 0; y < label->rows; ++y) {
        for (int x = 0; x < label->cols; ++x) {
            dst.at<uchar>(y, x) = label->at<uchar>(y, x) * step;
        }
    }
    imshow("Destination", dst);
    cv::waitKey(0);
    dst.release();
}

void save_image(cv:: Mat *label, int cluster_num) {
    int step = 256 / cluster_num;

    cv::Mat dst = cv::Mat::zeros(label->size(), CV_8U);
    for (int y = 0; y < label->rows; ++y) {
        for (int x = 0; x < label->cols; ++x) {
            dst.at<uchar>(y, x) = label->at<uchar>(y, x) * step;
        }
    }
    cv::imwrite("../result.png", dst);
    dst.release();
}

void print_mat(cv::Mat *image){
    for (int y = 0; y < image->rows; ++y) {
        for (int x = 0; x < image->cols; ++x) {
            printf("%d ", image->at<uchar>(y, x));
        }
        printf("\n");
    }
}
