#include "matrix.hpp"



// MATRIX FUNCTIONS
Matrix::Matrix(): num_rows(0), num_cols(0), data(nullptr) {}

Matrix::Matrix(int rows, int cols)
        : num_rows(rows), num_cols(cols), data(new double[num_rows*num_cols])
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> dist(4.0, 8.0);

            for(std::uint32_t i = 0; i < (num_rows * num_cols); i++)
            {
                data[i] = dist(gen);
            }
        }

Matrix::Matrix(const Matrix &m)
        {
            data = new double[m.num_rows * m.num_cols];

            this -> num_rows = m.num_rows;
            this -> num_cols = m.num_cols;

            for(std::uint32_t i = 0; i < (num_rows * num_cols); i++)
            {
                data[i] = m.data[i];
            }
        }

Matrix Matrix::hadamaard_product(const Matrix &m)
        {  
            if(!(this->num_rows == m.num_rows && this->num_cols == m.num_cols)) 
            {throw std::invalid_argument("DIMENSIONS OF THE MATRICES MUST MATCH!");}

            Matrix hadamard_matrix(this -> num_rows, this -> num_cols);

            for(std::uint32_t row = 0; row < this -> num_rows; row ++)
            {
                for(std::uint32_t col = 0; col < this -> num_cols; col ++)
                {
                    hadamard_matrix[row][col] = (*this)[row][col]*m[row][col];
                }
            }

            return hadamard_matrix;
        }

void Matrix::display_matrix()
{
    std::cout<<'\n';
    for(std::uint32_t row = 0; row < num_rows; row ++)
    {
        for(std::uint32_t col = 0; col < num_cols; col ++)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) 
                      << (*this)[row][col] << "  "; 
        }
        std::cout<<'\n';
    }
    std::cout<<'\n';

    LOG("MATRIX DIMENSIONS\n");

    std::cout<<"ROWS: "<<num_rows<<" COLS: "<<num_cols<<'\n';
}   

double Matrix::col_sum(Matrix &m, int col_index)
{
    double sum = 0.0;
    
    for(std::uint32_t row = 0; row < m.num_rows; row++)
    {
        sum += m[row][col_index];
    }

    return sum;
}

double Matrix::sum_of_elements(Matrix& m)
{
    double sum = 0;
    for(std::uint32_t row = 0; row < this->num_rows; row++)
    {
        for(std::uint32_t col = 0; col < this->num_cols; col++)
        {
            sum += m[row][col];
        }
    }

    return sum;
}


Matrix Matrix::collapse_horizontal(Matrix& m)
{
    Matrix result(1, m.num_cols);

    for(std::uint32_t col = 0; col < m.num_cols; col++)
    {
        result[1][col] = col_sum(m, col);
    }

    return result;
}

Matrix::~Matrix()
{
    delete[] data;
}


//MATRIX OPERATORS

const double* Matrix::operator[](int row) const
{
    return &data[row * num_cols];
}

double* Matrix::operator[](int row)
{
    return &data[row*num_cols];
}

Matrix& Matrix::operator=(const Matrix &m)
{

    if(this == &m)
    {
        return *this;
    }
    delete[] data;
    data = new double[m.num_rows * m.num_cols];

    this -> num_rows = m.num_rows;
    this -> num_cols = m.num_cols;

    for(std::uint32_t i = 0; i < (num_rows * num_cols); i++)
    {
        data[i] = m.data[i];
    }

    return *this;

}

Matrix Matrix::operator+(const Matrix &m) const
{
    if(!(this -> num_cols == m.num_cols && this -> num_rows == m.num_rows))
    {
        std::cerr<<"MATRIX DIMENSIONS DONT MATCH! "<<std::endl;
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    Matrix sum(this -> num_rows, this -> num_cols);
    
    for(std::uint32_t row = 0; row < num_rows; row ++)
    {
        for(std::uint32_t col = 0; col < num_cols; col ++)
        {
            sum.data[row*num_cols + col] = this->data[row*num_cols + col] + m.data[row*num_cols + col];
        }
    }

    return sum;
}

Matrix Matrix::operator~() const
{
    Matrix transpose(this->num_cols, this->num_rows);


    for(std::uint32_t row = 0; row < this->num_cols; row++)
    {
        for(std::uint32_t col = 0; col < this->num_rows; col++)
        {
            transpose[row][col] = (*this)[col][row];
        }
    }

    return transpose;
}

bool Matrix::operator==(const Matrix &m) const
{
    if(!(this->num_rows == m.num_rows && this->num_cols == m.num_cols)) return false;
    for(std::uint32_t row = 0; row < this->num_rows; row ++)
    {
        for(std::uint32_t col = 0; col < this->num_cols; col ++)
        {
            if(!((*this)[row][col] == m[row][col]))
            {
                return false;
            }
        }
    }
    return true;
}

Matrix Matrix::operator-(const Matrix &m) const
{
    if(!(this -> num_cols == m.num_cols && this -> num_rows == m.num_rows))
    {
        std::cerr<<"MATRIX DIMENSIONS DONT MATCH! "<<std::endl;
        throw std::invalid_argument("Matrix dimensions don't match");
    }
    Matrix diff(this -> num_rows, this -> num_cols);
    
    for(std::uint32_t row = 0; row < num_rows; row ++)
    {
        for(std::uint32_t col = 0; col < num_cols; col ++)
        {
            diff.data[row*num_cols + col] = this->data[row*num_cols + col] - m.data[row*num_cols + col];
        }
    }

    return diff;

}

Matrix Matrix::operator*(const Matrix &m) const
{
    if (!(this -> num_cols == m.num_rows))
    {
        throw std::invalid_argument("Dimensions dont match to multiply");
    }

    Matrix mult(this->num_rows, m.num_cols);

    for(std::uint32_t i = 0; i < this->num_rows; i++)
    {
        for(std::uint32_t j = 0; j < m.num_cols; j++)
        {
            double sum = 0.0;
            for(std::uint32_t k = 0; k < this->num_cols; k++)
            {
                sum += (*this)[i][k]*m[k][j];
            }

            mult[i][j] = sum;
        }
    }

    return mult;

}

Matrix Matrix::operator*(const double scalar) const
{

    Matrix result(this->num_rows, this->num_cols);

    for(std::uint32_t row = 0; row < this->num_rows; row++)
    {
        for(std::uint32_t col = 0; col < this->num_cols; col++)
        {
            result[row][col] = (*this)[row][col]*scalar; 
        }
    }

    return result;
}

std::size_t Matrix::num_elements()
{
    return this->num_rows * this->num_cols;
}