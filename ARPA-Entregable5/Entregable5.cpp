#include <mpi.h>
#include <iostream>

using namespace std;

constexpr int RANK_MASTER = 0;

/*
    Genera valores aleatorios para una matriz.

    @param matrix: matriz a la que se le asignar�n los valores aleatorios.
    @param rows: n�mero de filas de la matriz.
    @param cols: n�mero de columnas de la matriz.
*/
static void generateMatrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

/*
    Funci�n principal.

    @param argc: n�mero de argumentos.
    @param argv: argumentos pasados por l�nea de comandos.
    @return 0 si la ejecuci�n fue exitosa.
*/
int main(int argc, char **argv)
{
    // Tama�o de la Matriz
    int tamMatriz = 5;
    // Variables para MPI
    int rank, size = 0;
    // Variables para la divisi�n de las filas
    int rows_per_process, extra_rows, start_row, end_row, local_rows = 0;
    // Variables para la divisi�n de las filas
    int *recvcounts, *displs, *sendcounts_A = nullptr, *displs_A = nullptr;
    // Datos de las matrices
    float *A_data, *B_data, *C_data;
    // Matrices
    float **A, **B, **C;
    // Variables para medir el tiempo de ejecuci�n
    double start_time, end_time = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == RANK_MASTER)
    {
        if (argc > 1)
        {
            char *end;
            long val = strtol(argv[1], &end, 10);
            if (*end == '\0' && val > 0)
            {
                tamMatriz = static_cast<int>(val);
            }
            else
            {
                fprintf(stderr, "Error: El argumento debe ser un n�mero entero positivo.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            // Liberar memoria
            free(end);
        }
        else
        {
            printf("Ingrese el tama�o de las matrices cuadradas: ");
            while (!(cin >> tamMatriz) || tamMatriz <= 0)
            {
                cin.clear();
                cin.ignore(numeric_limits<streamsize>::max(), '\n');
                printf("Error: Ingrese un n�mero entero positivo: ");
            }
        }
    }

	start_time = MPI_Wtime();

    MPI_Bcast(&tamMatriz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcular el n�mero de filas por proceso
    rows_per_process = tamMatriz / size;
    extra_rows = tamMatriz % size;
    start_row = rank * rows_per_process + (rank < extra_rows ? rank : extra_rows);
    end_row = start_row + rows_per_process + (rank < extra_rows ? 1 : 0);
    local_rows = end_row - start_row;

    // Reservar memoria solo para las filas necesarias en cada proceso
    recvcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    A_data = (float *)malloc(static_cast<unsigned long long>(tamMatriz) * tamMatriz * sizeof(float));
    B_data = (float *)malloc(static_cast<unsigned long long>(tamMatriz) * tamMatriz * sizeof(float));
    C_data = (float *)malloc(static_cast<unsigned long long>(local_rows) * tamMatriz * sizeof(float));
    A = (float **)malloc(tamMatriz * sizeof(float *));
    B = (float **)malloc(tamMatriz * sizeof(float *));
    C = (float **)malloc(local_rows * sizeof(float *));

    // Asignar memoria para las filas de las matrices
    for (int i = 0; i < tamMatriz; i++)
    {
        A[i] = &A_data[i * tamMatriz];
        B[i] = &B_data[i * tamMatriz];
    }
    for (int i = 0; i < local_rows; i++)
    {
        C[i] = &C_data[i * tamMatriz];
    }

    if (rank == RANK_MASTER)
    {
        srand(time(NULL));
        generateMatrix(A, tamMatriz, tamMatriz);
        generateMatrix(B, tamMatriz, tamMatriz);
    }

    // Solo el proceso maestro prepara la matriz completa y los desplazamientos
    if (rank == RANK_MASTER)
    {
        sendcounts_A = (int *)malloc(size * sizeof(int));
        displs_A = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++)
        {
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
        MPI_COMM_WORLD);

    // Broadcast de B, necesaria completa en todos los procesos
    MPI_Bcast(B_data, tamMatriz * tamMatriz, MPI_FLOAT, RANK_MASTER, MPI_COMM_WORLD);

    // Liberar memoria en el maestro
    if (rank == RANK_MASTER)
    {
        free(sendcounts_A);
        free(displs_A);
    }

    // Multiplicar
    for (int i = 0; i < local_rows; i++)
    {
        for (int k = 0; k < tamMatriz; k++)
        {
            C[i][k] = 0.0f;
            for (int j = 0; j < tamMatriz; j++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    for (int i = 0; i < size; i++)
    {
        recvcounts[i] = (tamMatriz / size + (i < extra_rows ? 1 : 0)) * tamMatriz;
        displs[i] = (i * rows_per_process + (i < extra_rows ? i : extra_rows)) * tamMatriz;
    }

    MPI_Gatherv(C_data, local_rows * tamMatriz, MPI_FLOAT,
                A_data, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

	end_time = MPI_Wtime();

	if (rank == RANK_MASTER)
	{
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