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

    // Memoria dinámica para las matrices
    float** A, ** B, ** C;
    float* A_data, * B_data, * C_data;

    // Número de filas por proceso
    int rows_per_process = tamMatriz / size;
    // Número de filas adicionales para los primeros procesos
    int extra_rows = tamMatriz % size;
    // Rango de filas para el proceso actual
    int start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    int end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);
    int local_rows = end_row - start_row;

    // Reservar memoria para las matrices en posiciones contiguas
    if (rank == RANK_MASTER) {
        // El maestro necesita espacio para toda la matriz A
        A = (float**)malloc(tamMatriz * sizeof(float*));
        C = (float**)malloc(tamMatriz * sizeof(float*));
        A_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));
        C_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));
        for (int i = 0; i < tamMatriz; i++) {
            A[i] = &A_data[i * tamMatriz];
        }
    }
    else {
        // Los demás procesos sólo necesitan espacio para sus filas locales de A
        A = (float**)malloc(local_rows * sizeof(float*));
        C = (float**)malloc(local_rows * sizeof(float*));
        A_data = (float*)malloc(local_rows * tamMatriz * sizeof(float));
        C_data = (float*)malloc(local_rows * tamMatriz * sizeof(float));
        for (int i = 0; i < local_rows; i++) {
            A[i] = &A_data[i * tamMatriz];
            C[i] = &C_data[i * tamMatriz];
        }
    }

    // La matriz B es necesaria completa en todos los procesos
    B = (float**)malloc(tamMatriz * sizeof(float*));
    B_data = (float*)malloc(tamMatriz * tamMatriz * sizeof(float));
    for (int i = 0; i < tamMatriz; i++) {
        B[i] = &B_data[i * tamMatriz];
        if (B_data == nullptr) {
            fprintf(stderr, "Error: No se pudo asignar memoria para B_data.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < tamMatriz; i++) {
            B[i] = &B_data[i * tamMatriz];
        }
    }


    // Asignar memoria para las filas de las matrices
    for (int i = 0; i < local_rows; i++) {
        A[i] = &A_data[i * tamMatriz];
        C[i] = &C_data[i * tamMatriz];
    }
    for (int i = 0; i < tamMatriz; i++) {
        B[i] = &B_data[i * tamMatriz];
    }

    // Solo el proceso maestro prepara la matriz completa y los desplazamientos
    if (rank == RANK_MASTER) {
        srand(static_cast<unsigned int>(time(NULL)));
        generateMatrix(A, tamMatriz, tamMatriz);
        generateMatrix(B, tamMatriz, tamMatriz);
    }

    // Preparar los recuentos y desplazamientos para Scatterv
    int* sendcounts_A = nullptr;
    int* displs_A = nullptr;
    if (rank == RANK_MASTER) {
        sendcounts_A = (int*)malloc(size * sizeof(int));
        displs_A = (int*)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_process + (i < extra_rows ? 1 : 0);
            sendcounts_A[i] = rows * tamMatriz;
            displs_A[i] = offset;
            offset += sendcounts_A[i];
        }
    }

    // Distribuir las filas correspondientes de A a cada proceso
    MPI_Scatterv(
        rank == RANK_MASTER ? A_data : nullptr, // Enviar desde A_data en el maestro
        sendcounts_A,
        displs_A,
        MPI_FLOAT,
        A_data, // Recibir en A_data local
        local_rows * tamMatriz,
        MPI_FLOAT,
        RANK_MASTER,
        MPI_COMM_WORLD
    );

    // Broadcast de B, necesaria completa en todos los procesos
    MPI_Bcast(B_data, tamMatriz * tamMatriz, MPI_FLOAT, RANK_MASTER, MPI_COMM_WORLD);

    // Liberar memoria en el maestro
    if (rank == RANK_MASTER) {
        free(sendcounts_A);
        free(displs_A);
    }

    // Multiplicar
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < tamMatriz; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < tamMatriz; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Pillar resultados
    int* recvcounts = nullptr;
    int* displs = nullptr;
    if (rank == RANK_MASTER) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));
    }
    for (int i = 0; i < size; i++) {
        recvcounts[i] = (tamMatriz / size + (i < extra_rows ? 1 : 0)) * tamMatriz;
        displs[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * tamMatriz;
    }

    MPI_Gatherv(
        C_data, // Enviar desde C_data local
        local_rows* tamMatriz,
        MPI_FLOAT,
        rank == RANK_MASTER ? C_data : nullptr, // Recibir en C_data en el maestro
        recvcounts,
        displs,
        MPI_FLOAT,
        RANK_MASTER,
        MPI_COMM_WORLD
    );

    if (rank == RANK_MASTER) {
        double end_time = MPI_Wtime();
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