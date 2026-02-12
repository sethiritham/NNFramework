#include <cstdio>
#include <stdio.h>
#include <random>
#include <iostream>
#include <vector>
#include <stdexcept> 
#define LOG(x) std::cout<<x<<std::endl
#include <string>

class Matrix
{
    // Init variables
    public:
        int num_rows;
        int num_cols;
    
    private:
        double *data;

    // Class fucntions
    public:
        Matrix();
        
        Matrix(int rows, int cols);

        // Copy constructor
        Matrix(const Matrix &m);
        
        Matrix hadamaard_product(const Matrix &m);
        
        void display_matrix();

        static void fill_matrix_array(std::vector<double> values, Matrix &matrix)
        {
            if(!(values.size() == matrix.num_cols * matrix.num_rows))
            {
                throw std::invalid_argument("ERROR THE SIZE OF THE INPUT DOESNT MATCH THE DIMENSION OF THE MATRIX ");
            }
            
            for(int row = 0; row < matrix.num_rows; row++)
            {
                for(int col = 0 ;col < matrix.num_cols; col++)
                {
                    matrix[row][col] = values[row + col];
                }
            }
        }

        static void fill_matrix_double(double fill, Matrix &m)
        {
            for(int i = 0; i < m.num_rows; i++)
            {
                for(int j = 0; j < m.num_cols; j++)
                {
                    m[i][j] = fill;
                }
            }
        }

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
        }

        template<typename T, typename Func>
        auto map_matrix(const Matrix& m, Func& f)
        {
            for(int row = 0; row < m.num_rows; row++)
            {
                for(int col = 0; col < m.num_cols; col++)
                {
                    Func(m[row][col]);
                }
            }
        }

        double col_sum(Matrix &m, int col_index);

        Matrix collapse_horizontal(Matrix& m);

        ~Matrix();

    // Matrix operators
    public:
        const double *operator[](int row) const;

        double *operator[](int row);

        Matrix &operator=(const Matrix &m);

        Matrix operator+(const Matrix &m) const;

        Matrix operator~() const;

        bool operator==(const Matrix &m) const;

        Matrix operator-(const Matrix &m) const;

        Matrix operator*(const Matrix &m) const;
};
