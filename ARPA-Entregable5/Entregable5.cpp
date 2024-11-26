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
    Imprime una línea de la matriz

    @param tamMatrix: tamaño de la matriz
*/
static void printLine(int tamMatrix) {
    printf("+");
    for (int j = 0; j < tamMatrix + 1; ++j) {
        printf("-------");
    }
    printf("-----+\n");
}

/*
    Imprime una matriz

    @param matrix: matriz a imprimir
    @param rows: número de filas de la matriz
    @param cols: número de columnas de la matriz
    @param limit: límite de impresión
*/
static void printMatrix(float** matrix, int rows, int cols, int limit = 0) {
    if (limit == 0) {
        limit = rows;
    }

    // Imprimir la línea superior
    printLine(limit);

    for (int i = 0; i < rows && i < limit; ++i) {
        // Imprimir los valores de la fila
        printf("|");
        for (int j = 0; j < cols && j < limit; ++j) {
            printf(" %.2f |", matrix[i][j]);
        }
        if (cols > limit) {
            printf(" ... | %.2f |", matrix[i][cols - 1]);
        }
        printf("\n");

        // Imprimir las líneas intermedias e inferior de la fila
        printLine(limit);
    }

    if (rows > limit) {
        // Imprimir puntos suspensivos para las filas intermedias
        printf("|");
        for (int j = 0; j < limit; ++j) {
            printf(" ...  |");
        }
        if (cols > limit) {
            printf(" ... | ...  |");
        }
        printf("\n");

        // Imprimir la línea intermedia
        printLine(limit);

        // Imprimir la última fila
        printf("|");
        for (int j = 0; j < limit; ++j) {
            printf(" %.2f |", matrix[rows - 1][j]);
        }
        if (cols > limit) {
            printf(" ... | %.2f |", matrix[rows - 1][cols - 1]);
        }
        printf("\n");

        // Imprimir la línea inferior
        printLine(limit);
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

    // Memoria dinámica para las matrices
    float** A, ** B, ** C;
    float* A_data, * B_data, * C_data;

    // Reservar memoria para las matrices en posiciones contiguas
    A = (float**)malloc(tamMatriz * sizeof(float*));
    B = (float**)malloc(tamMatriz * sizeof(float*));
    C = (float**)malloc(tamMatriz * sizeof(float*));
    A_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));
    B_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));
    C_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));

    // Asignar memoria para las filas de las matrices
    for (int i = 0; i < tamMatriz; i++) {
        A[i] = &A_data[i * tamMatriz];
        B[i] = &B_data[i * tamMatriz];
        C[i] = &C_data[i * tamMatriz];
    }

    double start_generation_time = MPI_Wtime();

    if (rank == RANK_MASTER) {
        srand(time(NULL));
        generateMatrix(A, tamMatriz, tamMatriz);
        generateMatrix(B, tamMatriz, tamMatriz);
        printf("Matriz A:\n");
        printMatrix(A, tamMatriz, tamMatriz, 5);
        printf("Matriz B:\n");
        printMatrix(B, tamMatriz, tamMatriz, 5);
    }

    double end_generation_time = MPI_Wtime();

    // Broadcast matrices A and B
    MPI_Bcast(A_data, tamMatriz * tamMatriz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_data, tamMatriz * tamMatriz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Número de filas por proceso
    int rows_per_process = tamMatriz / size;
    // Número de filas adicionales para los primeros procesos
    int extra_rows = tamMatriz % size;
    // Rango de filas para el proceso actual
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);

    double start_multiplication_time = MPI_Wtime();

    // Multiplicar
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < tamMatriz; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < tamMatriz; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    double end_multiplication_time = MPI_Wtime();

    // Pillar resultados
    int* recvcounts = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (tamMatriz / size + (i < extra_rows ? 1 : 0)) * tamMatriz;
        displs[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * tamMatriz;
    }

    MPI_Gatherv(&C_data[start_row * tamMatriz], (end_row - start_row) * tamMatriz, MPI_FLOAT,
        C_data, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == RANK_MASTER) {
        // Mostrar porción de la matriz C
        printf("Matriz C:\n");
        printMatrix(C, tamMatriz, tamMatriz, 5);

        printf("Tiempo de ejecución global: %f segundos\n", end_time - start_time);
        printf("Tiempo de generación de matrices: %f segundos\n", end_generation_time - start_generation_time);
        printf("Tiempo de multiplicación de matrices: %f segundos\n", end_multiplication_time - start_multiplication_time);
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