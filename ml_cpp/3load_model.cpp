#include <torch/script.h>
#include "utils_torch.hpp"
#include "utils.hpp"


int main(int argc, char ** argv) {

	print_str(argv[0]);

	Net net;
	std::string nn_saved_path = "./net.pt";
	load_model(net, nn_saved_path);

	torch::optim::Adam optimizer(net->parameters());
	std::string optimizer_saved_path = "optimizer_state.pt";
	//torch::load(optimizer, optimizer_saved_path);
	load_optimizer_state(& optimizer, optimizer_saved_path);

	//optimizer.reset(net->parameters());

	// 训练数据集
	torch::Tensor x = torch::linspace(0, 3.14, 100).view({ 100, 1 }); // 
	torch::Tensor y = torch::sin(3. * x - 0.5);

	// Execute the model and turn its output into a tensor.
	torch::Tensor output = net(x);

	// 继续训练
	for (int epoch = 0; epoch < 10000; epoch++) {
		torch::Tensor y_pre = net(x);
		torch::Tensor mse = torch::mse_loss(y_pre, y);

		optimizer.zero_grad();
		mse.backward();
		optimizer.step();

		if (epoch % 100 == 0) {
			printf("Epoch %d \t loss %f\n", epoch, mse.cpu().item());
		}
	}
	
	save_model(net, nn_saved_path);
	save_optimizer_state(& optimizer, optimizer_saved_path);


	return 0;
}