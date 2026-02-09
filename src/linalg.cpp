#include <stdexcept>

#include "Tensor.h"
#include "linalg.h"

Tensor flatten(const Tensor& t, size_t start_dim = 1)
{
	size_t new_numel = 1;

	for (size_t i = start_dim; i < t.shape.size(); ++i)
	{
		new_numel *= t.shape[i];
	}

	std::vector<size_t> new_shape;
	for (size_t i = 0; i < start_dim; ++i)
	{
		new_shape.push_back(t.shape[i]);
	}
	new_shape.push_back(new_numel);

	Tensor flat(new_shape);

	flat.data = t.data;

	return flat;
}

Tensor matmul(const Tensor& A, const Tensor& B)
{
	size_t inner = A.shape[1];
	if (B.shape[0] != inner)
	{
		throw std::runtime_error("Matrix of dimensions does not match for matrix multiplication.");
	}

	size_t rows = A.shape[0];
	size_t cols = B.shape[1];

	Tensor result({ rows, cols });

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			float sum = 0.0f;
			
			for (size_t k = 0; k < inner; ++k)
			{
				sum += A({ i, k }) * B({ k, j });
			}

			result({ i, j }) = sum;
		}
	}

	return result;
}