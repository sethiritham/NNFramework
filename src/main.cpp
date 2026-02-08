#include "../math/matrix.hpp"
#include <random>

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(2,5);
    Matrix m(3, 3);
    Matrix m1(3, 3);

    for(int c = 0; c < 3; c++)
    {
        for(int r = 0; r < 3; r++)
        {
            m[r][c] = dist(gen);
            m1[r][c] = dist(gen);
        }
    }

    m.display_matrix();
    
    Matrix m2 = !m; 

    m2.display_matrix();
}