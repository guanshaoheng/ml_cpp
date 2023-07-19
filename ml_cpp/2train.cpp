#include <iostream>
#include "utils.hpp"
#include "utils_torch.hpp"

int main(int argc, char** argv) {

	print_str(argv[0]);

	// �趨�����
	torch::manual_seed(1);

	//����Ƿ����cuda
	torch::DeviceType device_cpu_type = torch::kCPU;
	torch::DeviceType device_CUDA_type = torch::kCUDA;
	torch::Device device(device_cpu_type);
	torch::Device device_cpu(device_cpu_type);
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Training on GPU." << std::endl;
		device = torch::Device(device_CUDA_type);
	}
	else {
		std::cout << "Training on CPU." << std::endl;
	}

	// ѵ�����ݼ�
	torch::Tensor x = torch::linspace(0, 3.14, 100).view({ 100, 1 }).to(device); // 
	torch::Tensor y = torch::sin(3. * x - 0.5);
	
	// ��ʼ������  //ע�⣺��������һ���������bug
	//�����������죬���ʹ��debug�汾���룬�ͻ�����⣬
	// ����ʹ��release�汾����Ͳ��������
	//��ĳ����ٵ�����������
	// std::shared_ptr<Net> net = std::make_shared<Net>(1, 1, 100); // ��������ָ�붨������
	Net net(1, 1, 100); 
	// Network net(1, 1, 100); // ��������
	(*net).to(device);

	//��ʼ���Ż���
	torch::optim::Adam  optimizer(net->parameters());

	for (int epoch = 0; epoch < 10000; epoch++) {

		torch::Tensor y_pre = net(x);
		torch::Tensor mse = torch::mse_loss(y_pre, y);

		optimizer.zero_grad();
		mse.backward();
		optimizer.step();	

		if (epoch % 100 == 0) {
			cout << "Epoch " << epoch
				<< " Curretn loss is:" << mse.cpu().item() << endl;
		}
	}
	
	//����ģ��
	string nn_save_path = "net.pt";
	save_model(net, nn_save_path);  // ע������Ӧ�ô��������ָ��

	//�����Ż���
	string optm_save_path = "optimizer_state.pt";
	save_optimizer_state(& optimizer, optm_save_path);

	return 0;

}