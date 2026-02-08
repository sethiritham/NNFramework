#include "../math/matrix.hpp"

int main()
{

    Matrix m(3, 3);
    Matrix m1(3, 3);

    for(int c = 0; c < 3; c++)
    {
        for(int r = 0; r < 3; r++)
        {
            m[r][c] = 1;
            m1[r][c] = 2;
        }
    }

    m.display_matrix();
    m1.display_matrix();

    Matrix m2 = m + m1;

    m2.display_matrix();
}