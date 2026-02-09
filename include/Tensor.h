#pragma once

#include <vector>

struct Tensor
{
	// Raw data.
	std::vector<float> data;

	// Tensor in the shape of e.g. {batch, channels, height, width}.
	std::vector<size_t> shape;
	
	// Constructor.
	explicit Tensor(const std::vector<size_t>& shape);

	// Total number of elements.
	size_t numel() const;

	// To be able to find indices in the tensor.
	float& operator()(std::initializer_list<size_t> idx);
};