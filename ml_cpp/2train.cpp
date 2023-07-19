#include <iostream>
#include "utils.hpp"
#include "utils_torch.hpp"

int main(int argc, char** argv) {

	print_str(argv[0]);

	// 设定随机数
	torch::manual_seed(1);

	//检查是否调用cuda
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

	// 训练数据集
	torch::Tensor x = torch::linspace(0, 3.14, 100).view({ 100, 1 }).to(device); // 
	torch::Tensor y = torch::sin(3. * x - 0.5);
	
	// 初始化网络  //注意：在这里有一个很逆天的bug
	//困扰了我两天，如果使用debug版本编译，就会出问题，
	// 但是使用release版本编译就不会出问题
	//真的超级操蛋。。。。。
	// std::shared_ptr<Net> net = std::make_shared<Net>(1, 1, 100); // 采用智能指针定义网络
	Net net(1, 1, 100); 
	// Network net(1, 1, 100); // 定义网络
	(*net).to(device);

	//初始化优化器
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
	
	//保存模型
	string nn_save_path = "net.pt";
	save_model(net, nn_save_path);  // 注意这里应该传入网络的指针

	//保存优化器
	string optm_save_path = "optimizer_state.pt";
	save_optimizer_state(& optimizer, optm_save_path);

	return 0;

}