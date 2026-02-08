#include <cstdio>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <stdexcept> 
#include <string>

class Matrix
{
    private:
        int r;
        int c;
        double *data;

    public:
        Matrix(int rows, int cols)
        {
            r = rows;
            c = cols;

            data = new double[r*c];

            for(int i = 0; i < (r * c); i++)
            {
                data[i] = 0.0;
            }
        }

        const double *operator[](int row) const
        {
            return &data[row * c];
        }

        double *operator[](int row)
        {
            return &data[row*c];
        }

        Matrix &operator=(const Matrix &m)
        {

            if(this == &m)
            {
                return *this;
            }
            delete[] data;
            data = new double[m.r * m.c];

            this -> r = m.r;
            this -> c = m.c;

            for(int i = 0; i < (r * c); i++)
            {
                data[i] = m.data[i];
            }

            return *this;

        }

        Matrix(const Matrix &m)
        {
            data = new double[m.r * m.c];

            this -> r = m.r;
            this -> c = m.c;

            for(int i = 0; i < (r * c); i++)
            {
                data[i] = m.data[i];
            }
        }

        Matrix operator+(const Matrix &m) const
        {
            if(!(this -> c == m.c && this -> r == m.r))
            {
                std::cerr<<"MATRIX DIMENSIONS DONT MATCH! "<<std::endl;
                throw std::invalid_argument("Matrix dimensions don't match");
            }
            Matrix sum(this -> r, this -> c);
            
            for(int row = 0; row < r; row ++)
            {
                for(int col = 0; col < c; col ++)
                {
                    sum.data[row*c + col] = this->data[row*c + col] + m.data[row*c + col];
                }
            }

            return sum;
        }
        void display_matrix()
        {
            for(int row = 0; row < r; row ++)
            {
                for(int col = 0; col < c; col ++)
                {
                    std::cout<<data[row * c + col]<<" ";
                }
                std::cout<<std::endl;
            }
        }

        ~Matrix()
        {
            delete[] data;
        }
};

