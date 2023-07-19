#include <torch/torch.h>
#include <iostream>


// ����һ����ȫ���Ӳ���ɵ�����

struct Net : torch::nn::Module {
    Net() {
        // ���첢��ע������ṹ�в�
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }
    // ʵ��ǰ�򴫲�
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x.reshape({ x.size(0),784 })));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{ nullptr }, fc3{ nullptr };
};

int main() {
    // ����һ���µ�����
    auto net = std::make_shared<Net>();

    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data").map(
            torch::data::transforms::Stack<>()),
        /*batch_size=*/64);

    torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

    for (size_t epoch = 1; epoch <= 10; ++epoch) {
        size_t batch_index = 0;
        // Iterate the data loader to yield batches from the dataset.
        for (auto& batch : *data_loader) {
            // �����ݶ�
            optimizer.zero_grad();
            // ����������������ģ��
            torch::Tensor prediction = net->forward(batch.data);
            // ����ģ��Ԥ��ֵ����ʵֵ֮������������ʧֵ
            torch::Tensor loss = torch::nll_loss(prediction, batch.target);
            //�ڷ��򴫲��м���������ݶ�
            loss.backward();

            // ���ڼ���õ��ݶ������²���
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                    << " | Loss: " << loss.item<float>() << std::endl;
                // ����ģ��
                torch::save(net, "net.pt");
            }
        }
    }
}