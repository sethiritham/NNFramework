#include <cstdio>
#include <vector>
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

        double *operator[](const int row)
        {
            return &data[row * c];
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

        ~Matrix()
        {
            delete[] data;
        }
};

