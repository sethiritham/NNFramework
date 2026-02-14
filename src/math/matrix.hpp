#include <cstdio>
#include <stdio.h>
#include <random>
#include <iomanip>
#include <iostream>
#include <vector>
#include <stdexcept> 
#define LOG(x) std::cout<<x<<std::endl
#include <string>

class Matrix
{
    // Init variables
    public:
        std::uint32_t num_rows;
        std::uint32_t num_cols;
    
    private:
        double *data;

    // Class fucntions
    public:
        /**
         * @brief Initialises matrix with 0 rows, 0 cols and no data
         */
        Matrix();
        
        /**
         * @brief Initialises matrix with given rows and cols and random data
         * @param rows number of rows 
         * @param cols number of cols
         */
        Matrix(int rows, int cols);

        // Copy constructor
        Matrix(const Matrix &m);
        
        /**
         * @brief element wise multiplicaton of 2 matrices
         */
        Matrix hadamaard_product(const Matrix &m);
        
        /**
         * @brief displays matrix
         */
        void display_matrix();

        /**
         * @brief returns column sum of a given column index
         */
        double col_sum(Matrix &m, int col_index);

        /**
         * @brief sum of all the elements in the Matrix
         */
        double sum_of_elements(Matrix& m);

        /**
         * @brief returns the number of elements in the matrix
         */
        std::size_t num_elements();

        /**
         * @brief collapes the matrix vertically to give a row matrix
         * all the elements of a given columns are added to return 
         */
        Matrix collapse_horizontal(Matrix& m);

        /**
         * @param data pointer freed
         */
        ~Matrix();

    public:
        //Static functions

        /**
         * @brief Matrix filled with a given value 
         */
        static void fill_matrix_array(std::vector<double> values, Matrix &matrix)
        {
            if(!(values.size() == matrix.num_elements()))
            {
                throw std::invalid_argument("ERROR THE SIZE OF THE INPUT DOESNT MATCH THE DIMENSION OF THE MATRIX ");
            }

            for(std::uint32_t row = 0; row < matrix.num_rows; row++)
            {
                for(std::uint32_t col = 0 ;col < matrix.num_cols; col++)
                {
                    matrix[row][col] = values[row + col];
                }
            }
        }

        /**
         * @param fill the value filled at each position in matrix
         */
        static void fill_matrix_double(double fill, Matrix &m)
        {
            for(std::uint32_t i = 0; i < m.num_rows; i++)
            {
                for(std::uint32_t j = 0; j < m.num_cols; j++)
                {
                    m[i][j] = fill;
                }
            }
        }

        /**
         * @param dim the dimension of Identity matrix
         */
        static Matrix identity(int dim)
        {
            Matrix iden_matrix(dim, dim);

            for(int i = 0; i < dim; i++)
            {
                for(int j = 0; j < dim; j++)
                {
                    if(i == j)
                    {
                        iden_matrix[i][j] = 1.0;
                    }
                    else
                    {
                        iden_matrix[i][j] = 0.0;
                    }
                }
            }

            return iden_matrix;
        }

        static Matrix row_wise_sum(const Matrix &matrix, const Matrix &row_matrix)
        {
            if(!(matrix.num_cols == row_matrix.num_cols))
            {
                return {};
            }

            Matrix result(matrix.num_rows, matrix.num_cols);

            for(std::uint32_t row = 0; row < matrix.num_rows; row++)
            {
                for(std::uint32_t col = 0; col < matrix.num_cols; col++)
                {
                    result[row][col] = matrix[row][col] + row_matrix[0][col];
                }
            }

            return result;
        }


    // Matrix operators
    public:
        /**
         * @brief read-only access for matrix data : matrix[row][col]
         */
        const double *operator[](int row) const;

        /**
         * @brief read and write access for matrix data : matrix[row][col]
         * @return double
         */
        double *operator[](int row);
        
        /**
         * @brief assigns data of x matrix to why in : x = y
         * @return Matrix
         */
        Matrix &operator=(const Matrix &m);

        /**
         * @brief matrix addition 
         * @return Matrix 
         */
        Matrix operator+(const Matrix &m) const;

        /**
         * @brief transpose
         * @return Matrix
         */
        Matrix operator~() const;

        /**
         * @brief checks for equality of matrix
         * @return bool
         */
        bool operator==(const Matrix &m) const;

        /**
         * @brief matrix subtraction
         * @return Matrix
         */
        Matrix operator-(const Matrix &m) const;

        /**
         * @brief matrix multiplication
         * @return Matrix
         */
        Matrix operator*(const Matrix &m) const;
        
        /**
         * @brief scalar multiplication on matrix 
         * @return matrix
         */
        Matrix operator*(const double scalar) const;
};

/**
 * @brief Given function operated on every matrix elements
 * @param f function to map over matrix
 */
template<typename T, typename Func>
Matrix map_matrix(const Matrix& m, Func f)
{
    Matrix result(m.num_rows, m.num_cols);

    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        for(std::uint32_t col = 0; col < m.num_cols; col++)
        {
            result[row][col] = f(m[row][col]);
        }
    }

    return result;
}