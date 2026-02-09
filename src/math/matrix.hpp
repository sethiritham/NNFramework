#include <cstdio>
#include <stdio.h>
#include <random>
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

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(4.0, 8.0);

            for(int i = 0; i < (r * c); i++)
            {
                data[i] = dist(gen);
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

        Matrix operator!() const
        {
            Matrix transpose(this->c, this->r);


            for(int row = 0; row < this->c; row++)
            {
                for(int col = 0; col < this->r; col++)
                {
                    transpose[row][col] = (*this)[col][row];
                }
            }

            return transpose;
        }

        bool operator==(const Matrix &m) const
        {
            if(!(this->r == m.r && this->c == m.c)) return false;
            for(int row = 0; row < this->r; row ++)
            {
                for(int col = 0; col < this->c; col ++)
                {
                    if(!((*this)[row][col] == m[row][col]))
                    {
                        return false;
                    }
                }
            }

            return true;
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

        Matrix operator>(const Matrix &m)
        {  
            if(!(this->r == m.r && this->c == m.c)) 
            {throw std::invalid_argument("DIMENSIONS OF THE MATRICES MUST MATCH!");}

            Matrix hadamard_matrix(this -> r, this -> c);

            for(int row = 0; row < this -> r; row ++)
            {
                for(int col = 0; col < this -> c; col ++)
                {
                    hadamard_matrix[row][col] = (*this)[row][col]*m[row][col];
                }
            }

            return hadamard_matrix;
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

        template<typename T, typename Func>
        auto map_matrix(const Matrix& m, Func f)
        {
            Matrix result_matrix(m.r, m.c);

            for(int row = 0; row < m.r; row++)
            {
                for(int col = 0; col < m.c; col++)
                {
                    result_matrix[row][col] = Func(m[row][col])
                }
            }

            return result_matrix;
        }

        ~Matrix()
        {
            delete[] data;
        }
};
