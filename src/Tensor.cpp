#include <vector>
#include <stdexcept>

#include "Tensor.h"

Tensor::Tensor(const std::vector<size_t>& shape_)
	: shape(shape_)
{
	size_t size = 1;
	for (size_t d : shape)
	{
		// The size will be the product of the dimensions of the shape.
		size *= d;
	}

	// Alocate memory to store the tensor and initialize it with zeros.
	data.resize(size, 0.0f);
}

size_t Tensor::numel() const
{
	return data.size();
}

float& Tensor::operator()(std::initializer_list<size_t> idx)
{
	std::vector<size_t> indices(idx);

	if (indices.size() != shape.size())
	{
		throw std::out_of_range("Tensor: wrong number of indices.");
	}

	size_t flat = 0;
	size_t stride = 1;

	auto i = indices.end();
	auto s = shape.end();
	while (i != indices.begin())
	{
		--i; --s;
		flat += (*i) * stride;
		stride *= (*s);
	}

	if (flat >= data.size())
	{
		throw std::out_of_range("Tensor: index out of bounds.");
	}

	return data[flat];
}