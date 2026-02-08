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

        Matrix operator-(const Matrix &m) const
        {
            if(!(this -> c == m.c && this -> r == m.r))
            {
                std::cerr<<"MATRIX DIMENSIONS DONT MATCH! "<<std::endl;
                throw std::invalid_argument("Matrix dimensions don't match");
            }
            Matrix diff(this -> r, this -> c);
            
            for(int row = 0; row < r; row ++)
            {
                for(int col = 0; col < c; col ++)
                {
                    diff.data[row*c + col] = this->data[row*c + col] - m.data[row*c + col];
                }
            }

            return diff;

        }

        Matrix operator*(const Matrix &m) const
        {
            if (!(this -> c == m.r))
            {
                throw std::invalid_argument("Dimensions dont match to multiply");
            }

            Matrix mult(this->r, m.c);

            for(int i = 0; i < this->r; i++)
            {
                for(int j = 0; j < m.c; j++)
                {
                    double sum = 0.0;
                    for(int k = 0; k < this->c; k++)
                    {
                        sum += (*this)[i][k]*m[k][j];
                    }

                    mult[i][j] = sum;
                }
            }

            return mult;

        }
        void display_matrix()
        {
            std::cout<<std::endl;
            for(int row = 0; row < r; row ++)
            {
                for(int col = 0; col < c; col ++)
                {
                    std::cout<<data[row * c + col]<<" ";
                }
                std::cout<<std::endl;
            }

            std::cout<<std::endl;
        }   

        ~Matrix()
        {
            delete[] data;
        }
};

