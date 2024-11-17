#include <iostream>
#include <fstream>
#include <mpi.h>

// Default values for simulation settings
int gridHeight = 5;
int gridWidth = 5;
double alpha = 0.1;
double beta = 0.3;
int omega = 2;
int numDays = 5;

class Person {
public:
    // Use int64_t to handle much larger values safely
    int64_t state;  // Higher bits for was_infected, lower bits for sick_days
    Person() : state(0) {}
};

void readSettingsFromFile(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open settings file.\n";
        return;
    }

    char label[50];
    if (file >> label >> gridHeight >>
        label >> gridWidth >>
        label >> alpha >>
        label >> beta >>
        label >> omega >>
        label >> numDays) {
    }
    else {
        std::cerr << "Error reading settings file.\n";
    }
    file.close();
}

unsigned int customRand(unsigned int& seed) {
    seed = seed * 1103515245 + 12345;
    return (seed / 65536) % 32768;
}

void initializeLocalGrid(Person* localGrid, int localRows, int totalRows, int rank, int numProcs) {
    int64_t totalPeople = static_cast<int64_t>(gridHeight) * gridWidth;  // Use int64_t to prevent overflow
    int64_t infectedCount = static_cast<int64_t>(alpha * totalPeople);
    unsigned int seed = 123456789 + rank;

    // Calculate global row range for this process
    int startRow = rank * (totalRows / numProcs);
    int endRow = (rank == numProcs - 1) ? totalRows : (rank + 1) * (totalRows / numProcs);
    localRows = endRow - startRow;

    // Initialize infected individuals
    int64_t localInfectedCount = infectedCount / numProcs;
    if (rank == 0) localInfectedCount += infectedCount % numProcs;

    for (int64_t count = 0; count < localInfectedCount; ++count) {
        int64_t idx;
        do {
            idx = customRand(seed) % (static_cast<int64_t>(localRows) * gridWidth);
        } while (localGrid[idx].state & 0x10000); // Check if already infected using higher bit

        localGrid[idx].state = 0x10001; // Set infected flag (bit 16) and sick_days=1
    }
}

void updateLocalGrid(Person* localGrid, Person* newLocalGrid, int localRows,
    Person* topRow, Person* bottomRow, int rank, int numProcs) {
    // Create a custom MPI datatype for Person
    MPI_Datatype person_type;
    MPI_Type_contiguous(sizeof(Person), MPI_BYTE, &person_type);
    MPI_Type_commit(&person_type);

    // Exchange boundary data with neighbors
    int topNeighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottomNeighbor = (rank == numProcs - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Request requests[4];
    MPI_Isend(localGrid, gridWidth, person_type, topNeighbor, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(localGrid + (localRows - 1) * gridWidth, gridWidth, person_type, bottomNeighbor, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Irecv(topRow, gridWidth, person_type, topNeighbor, 1, MPI_COMM_WORLD, &requests[2]);
    MPI_Irecv(bottomRow, gridWidth, person_type, bottomNeighbor, 0, MPI_COMM_WORLD, &requests[3]);

    // Process middle rows while communication is happening
    for (int i = 1; i < localRows - 1; ++i) {
        for (int j = 0; j < gridWidth; ++j) {
            int64_t idx = static_cast<int64_t>(i) * gridWidth + j;
            newLocalGrid[idx] = localGrid[idx];

            if (localGrid[idx].state & 0xFFFF) { // Check sick days (lower 16 bits)
                int sick_days = (localGrid[idx].state & 0xFFFF) + 1;
                newLocalGrid[idx].state = (sick_days >= omega) ? 0x10000 : (0x10000 | sick_days);
            }
            else if (!(localGrid[idx].state & 0x10000)) { // Not previously infected
                int sickNeighbors = 0;
                // Check left and right
                if (j > 0 && (localGrid[idx - 1].state & 0xFFFF)) sickNeighbors++;
                if (j < gridWidth - 1 && (localGrid[idx + 1].state & 0xFFFF)) sickNeighbors++;
                // Check above and below
                if (i > 0 && (localGrid[idx - gridWidth].state & 0xFFFF)) sickNeighbors++;
                if (i < localRows - 1 && (localGrid[idx + gridWidth].state & 0xFFFF)) sickNeighbors++;

                unsigned int seed = 123456789 + rank + i * gridWidth + j;
                if (customRand(seed) % 1000 < beta * sickNeighbors * 1000.0) {
                    newLocalGrid[idx].state = 0x10001;
                }
            }
        }
    }

    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
    MPI_Type_free(&person_type);

    // Process boundary rows
    if (localRows > 1) {
        for (int i = 0; i < localRows; i += localRows - 1) {
            for (int j = 0; j < gridWidth; ++j) {
                int64_t idx = static_cast<int64_t>(i) * gridWidth + j;
                newLocalGrid[idx] = localGrid[idx];

                if (localGrid[idx].state & 0xFFFF) {
                    int sick_days = (localGrid[idx].state & 0xFFFF) + 1;
                    newLocalGrid[idx].state = (sick_days >= omega) ? 0x10000 : (0x10000 | sick_days);
                }
                else if (!(localGrid[idx].state & 0x10000)) {
                    int sickNeighbors = 0;

                    if (j > 0 && (localGrid[idx - 1].state & 0xFFFF)) sickNeighbors++;
                    if (j < gridWidth - 1 && (localGrid[idx + 1].state & 0xFFFF)) sickNeighbors++;

                    if (i == 0) {
                        if (rank > 0 && (topRow[j].state & 0xFFFF)) sickNeighbors++;
                        if ((localGrid[idx + gridWidth].state & 0xFFFF)) sickNeighbors++;
                    }
                    else {
                        if ((localGrid[idx - gridWidth].state & 0xFFFF)) sickNeighbors++;
                        if (rank < numProcs - 1 && (bottomRow[j].state & 0xFFFF)) sickNeighbors++;
                    }

                    unsigned int seed = 123456789 + rank + i * gridWidth + j;
                    if (customRand(seed) % 1000 < beta * sickNeighbors * 1000.0) {
                        newLocalGrid[idx].state = 0x10001;
                    }
                }
            }
        }
    }
}

void writeGridToFile(std::ofstream& file, const Person* grid, int rows, int cols) {
    char* buffer = new char[cols * 2 + 2];

    for (int i = 0; i < rows; ++i) {
        int pos = 0;
        for (int j = 0; j < cols; ++j) {
            buffer[pos++] = (grid[i * cols + j].state & 0xFFFF) ? '1' : '0';
            buffer[pos++] = ' ';
        }
        buffer[pos++] = '\n';
        buffer[pos] = '\0';
        file.write(buffer, pos);
    }

    delete[] buffer;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (rank == 0) {
        readSettingsFromFile("settings.txt");
    }

    MPI_Bcast(&gridHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gridWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&omega, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numDays, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localRows = std::max(1, gridHeight / numProcs);
    if (rank == numProcs - 1) {
        localRows = gridHeight - (numProcs - 1) * localRows;
    }

    Person* localGrid = new Person[(localRows + 2) * gridWidth];
    Person* newLocalGrid = new Person[(localRows + 2) * gridWidth];
    Person* topRow = new Person[gridWidth];
    Person* bottomRow = new Person[gridWidth];

    initializeLocalGrid(localGrid + gridWidth, localRows, gridHeight, rank, numProcs);

    // Create a custom MPI datatype for Person
    MPI_Datatype person_type;
    MPI_Type_contiguous(sizeof(Person), MPI_BYTE, &person_type);
    MPI_Type_commit(&person_type);

    std::ofstream outFile;
    if (rank == 0) {
        outFile.open("flu_simulation.txt", std::ios::trunc);
        outFile << "Day 0:\n";
        writeGridToFile(outFile, localGrid + gridWidth, localRows, gridWidth);

        Person* recvBuffer = new Person[gridWidth * (gridHeight / numProcs + 2)];
        for (int p = 1; p < numProcs; ++p) {
            int rowsToReceive = (p == numProcs - 1) ?
                gridHeight - (numProcs - 1) * (gridHeight / numProcs) :
                gridHeight / numProcs;

            MPI_Recv(recvBuffer, rowsToReceive * gridWidth, person_type, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            writeGridToFile(outFile, recvBuffer, rowsToReceive, gridWidth);
        }
        delete[] recvBuffer;
    }
    else {
        MPI_Send(localGrid + gridWidth, localRows * gridWidth, person_type, 0, 0, MPI_COMM_WORLD);
    }

    double startTime = MPI_Wtime();

    for (int day = 1; day <= numDays; ++day) {
        updateLocalGrid(localGrid + gridWidth, newLocalGrid + gridWidth, localRows, topRow, bottomRow, rank, numProcs);

        if (rank == 0) {
            outFile << "Day " << day << ":\n";
            writeGridToFile(outFile, newLocalGrid + gridWidth, localRows, gridWidth);

            Person* recvBuffer = new Person[gridWidth * (gridHeight / numProcs + 2)];
            for (int p = 1; p < numProcs; ++p) {
                int rowsToReceive = (p == numProcs - 1) ?
                    gridHeight - (numProcs - 1) * (gridHeight / numProcs) :
                    gridHeight / numProcs;

                MPI_Recv(recvBuffer, rowsToReceive * gridWidth, person_type, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                writeGridToFile(outFile, recvBuffer, rowsToReceive, gridWidth);
            }
            delete[] recvBuffer;
        }
        else {
            MPI_Send(newLocalGrid + gridWidth, localRows * gridWidth, person_type, 0, 0, MPI_COMM_WORLD);
        }

        std::swap(localGrid, newLocalGrid);
    }

    MPI_Type_free(&person_type);

    if (rank == 0) {
        outFile.close();
        double endTime = MPI_Wtime();
        std::cout << "Simulation completed in: " << (endTime - startTime) << " seconds." << std::endl;
    }

    delete[] localGrid;
    delete[] newLocalGrid;
    delete[] topRow;
    delete[] bottomRow;

    MPI_Finalize();
    return 0;
}