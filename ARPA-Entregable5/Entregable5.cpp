#include <mpi.h>
#include <iostream>

using namespace std;

constexpr int RANK_MASTER = 0;

/*
    Genera valores aleatorios para una matriz

    @param matrix: matriz a la que se le asignarán los valores aleatorios
    @param rows: número de filas de la matriz
    @param cols: número de columnas de la matriz
*/
static void generateMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

/*
    Función principal

    @param argc: número de argumentos
    @param argv: argumentos
*/
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int tamMatriz = 5;
    if (rank == RANK_MASTER) {
        if (argc > 1) {
            char* end;
            long val = strtol(argv[1], &end, 10);
            if (*end == '\0' && val > 0) {
                tamMatriz = static_cast<int>(val);
            }
            else {
                fprintf(stderr, "Error: El argumento debe ser un número entero positivo.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            printf("Ingrese el tamaño de las matrices cuadradas: ");
            while (!(cin >> tamMatriz) || tamMatriz <= 0) {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                printf("Error: Ingrese un número entero positivo: ");
            }
        }
    }

    double start_time = MPI_Wtime();

    MPI_Bcast(&tamMatriz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcular el número de filas por proceso
    int rows_per_process = tamMatriz / size;
    int extra_rows = tamMatriz % size;
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);
    int local_rows = end_row - start_row;

    // Memoria dinámica para las matrices
    float** A, ** B, ** C;
    float* A_data, * B_data, * C_data;

    // Reservar memoria solo para las filas necesarias en cada proceso
    A = (float**)malloc(tamMatriz * sizeof(float*));
    B = (float**)malloc(tamMatriz * sizeof(float*));
    C = (float**)malloc(local_rows * sizeof(float*));
    A_data = (float*)malloc(static_cast<unsigned long long>(tamMatriz) * tamMatriz * sizeof(float));
    B_data = (float*)malloc(static_cast<unsigned long long>(tamMatriz) * tamMatriz * sizeof(float));
    C_data = (float*)malloc(static_cast<unsigned long long>(local_rows) * tamMatriz * sizeof(float));

    // Asignar memoria para las filas de las matrices
    for (int i = 0; i < tamMatriz; i++) {
        A[i] = &A_data[i * tamMatriz];
        B[i] = &B_data[i * tamMatriz];
    }
    for (int i = 0; i < local_rows; i++) {
        C[i] = &C_data[i * tamMatriz];
    }

    if (rank == RANK_MASTER) {
        srand(time(NULL));
        generateMatrix(A, tamMatriz, tamMatriz);
        generateMatrix(B, tamMatriz, tamMatriz);
    }

    // Broadcast matrices A and B
    MPI_Bcast(A_data, tamMatriz * tamMatriz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_data, tamMatriz * tamMatriz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Multiplicar (optimizado para mejor uso de caché)
    for (int i = 0; i < local_rows; i++) {
        for (int k = 0; k < tamMatriz; k++) {
            for (int j = 0; j < tamMatriz; j++) {
                C[i][j] += A[i + start_row][k] * B[k][j];
            }
        }
    }

    // Recoger resultados
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (tamMatriz / size + (i < extra_rows ? 1 : 0)) * tamMatriz;
        displs[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * tamMatriz;
    }

    MPI_Gatherv(C_data, local_rows * tamMatriz, MPI_FLOAT,
        A_data, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == RANK_MASTER) {
        printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
    }

    // Liberar memoria
    free(A_data);
    free(B_data);
    free(C_data);
    free(A);
    free(B);
    free(C);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}