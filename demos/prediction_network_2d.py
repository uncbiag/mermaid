import torch
import torch.nn as nn
import torch.nn.functional as F


class encoder_block(nn.Module):
	def __init__(self, input_feature, output_feature, use_dropout):
		super(encoder_block, self).__init__()
		self.conv_input = nn.Conv2d(input_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock1 = nn.Conv2d(output_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock2 = nn.Conv2d(output_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_pooling = nn.Conv2d(output_feature, output_feature, 2, 2, 1, 1).cuda()
		self.prelu1 = nn.PReLU().cuda()
		self.prelu2 = nn.PReLU().cuda()
		self.prelu3 = nn.PReLU().cuda()
		self.prelu4 = nn.PReLU().cuda()
		self.use_dropout = use_dropout;
		self.dropout = nn.Dropout(0.2).cuda()
	def apply_dropout(self, input):
		if self.use_dropout:
			return self.dropout(input)
		else:
			return input;
	def forward(self, x):
		output = self.conv_input(x)
		output = self.apply_dropout(self.prelu1(output));
		output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)));
		output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)));
		return self.prelu4(self.conv_pooling(output));


class decoder_block(nn.Module):
	def __init__(self, input_feature, output_feature, pooling_filter, use_dropout):
		super(decoder_block, self).__init__()
		self.conv_unpooling = nn.ConvTranspose2d(input_feature, input_feature, pooling_filter, 2, 1).cuda()
		self.conv_inblock1 = nn.Conv2d(input_feature, input_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock2 = nn.Conv2d(input_feature, input_feature, 3, 1, 1, 1).cuda()
		self.conv_output = nn.Conv2d(input_feature, output_feature, 3, 1, 1, 1).cuda()
		self.prelu1 = nn.PReLU().cuda()
		self.prelu2 = nn.PReLU().cuda()
		self.prelu3 = nn.PReLU().cuda()
		self.prelu4 = nn.PReLU().cuda()
		self.use_dropout = use_dropout;
		self.dropout = nn.Dropout(0.2).cuda()
		self.output_feature = output_feature;
	def apply_dropout(self, input):
		if self.use_dropout:
			return self.dropout(input);
		else:
			return input;
	def forward(self, x):
		output = self.prelu1(self.conv_unpooling(x));
		output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)));	
		output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)));
		if self.output_feature == 1: # generates final momentum
			return self.conv_output(output);
		else: # generates intermediate results
			return self.apply_dropout(self.prelu4(self.conv_output(output)));


class net(nn.Module):
	def __init__(self, feature_num, use_dropout = False):
		super(net, self).__init__()
		self.encoder_m_1 = encoder_block(1, feature_num, use_dropout);
		self.encoder_t_1 = encoder_block(1, feature_num, use_dropout);
		self.encoder_m_2 = encoder_block(feature_num, feature_num*2, use_dropout)
		self.encoder_t_2 = encoder_block(feature_num, feature_num*2, use_dropout)

		self.decoder_x_1 = decoder_block(feature_num * 4, feature_num * 2, 2, use_dropout);
		self.decoder_y_1 = decoder_block(feature_num * 4, feature_num * 2, 2, use_dropout);

		self.decoder_x_2 = decoder_block(feature_num*2, 1, 3, use_dropout);
		self.decoder_y_2 = decoder_block(feature_num*2, 1, 3, use_dropout);

	def forward(self, x):
		[moving, target] = torch.split(x, 1, 1);
		# print(self.decoder2)

		moving_encoder_output = self.encoder_m_2(self.encoder_m_1(moving));
		target_encoder_output = self.encoder_t_2(self.encoder_t_1(target));
		combine_encoder_output = torch.cat((moving_encoder_output, target_encoder_output), 1);
		predict_result_x = self.decoder_x_2(self.decoder_x_1(combine_encoder_output))
		predict_result_y = self.decoder_y_2(self.decoder_y_1(combine_encoder_output))


		#print(predict_result)
		return torch.cat((predict_result_x, predict_result_y), 1);

