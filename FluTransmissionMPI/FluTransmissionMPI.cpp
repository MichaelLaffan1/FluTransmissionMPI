#include <mpi.h>
#include <iostream>
#include <fstream>

// Parameters with default values
int ROWS;
int COLS;
double ALPHA;
double BETA;
int OMEGA;
int SIMULATION_DAYS;

struct Person {
    bool sick;
    int sick_days;
    bool recovered;
    Person() : sick(false), sick_days(0), recovered(false) {}
};

unsigned int simpleRand(unsigned int seed) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed;
}

// Read parameters from settings.txt
bool readSettings(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening " << filename << "\n";
        return false;
    }
    char label[50];
    if (file >> label >> ROWS &&
        file >> label >> COLS &&
        file >> label >> ALPHA &&
        file >> label >> BETA &&
        file >> label >> OMEGA &&
        file >> label >> SIMULATION_DAYS) {
        std::cout << "Values read in successfully.\n";
        
    }
    else {
        std::cerr << "Error reading settings file.\n";
    }
    file.close();
    return true;
}

// Initialize grid with a percentage of people sick on day 0
void initializeGrid(Person** grid, int rows, unsigned int& seed) {
    int total_population = rows * COLS;
    int sick_people = static_cast<int>(total_population * ALPHA);
    int initialized = 0;

    for (int i = 0; i < rows && initialized < sick_people; ++i) {
        for (int j = 0; j < COLS && initialized < sick_people; ++j) {
            seed = simpleRand(seed);
            if ((seed % 100) < (ALPHA * 100)) {
                grid[i][j].sick = true;
                grid[i][j].sick_days = 1;
                initialized++;
            }
        }
    }
}

// Simulate one day of infection spread and recovery
void simulateDay(Person** grid, Person** next_grid, int start_row, int end_row, unsigned int& seed) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < COLS; ++j) {
            next_grid[i][j] = grid[i][j];

            if (grid[i][j].sick) {
                next_grid[i][j].sick_days++;
                if (next_grid[i][j].sick_days > OMEGA) {
                    next_grid[i][j].sick = false;
                    next_grid[i][j].recovered = true;
                    next_grid[i][j].sick_days = 0;
                }
            }
            else if (!grid[i][j].recovered) {
                int neighbors_sick = 0;
                if (i > 0 && grid[i - 1][j].sick) neighbors_sick++;
                if (i < ROWS - 1 && grid[i + 1][j].sick) neighbors_sick++;
                if (j > 0 && grid[i][j - 1].sick) neighbors_sick++;
                if (j < COLS - 1 && grid[i][j + 1].sick) neighbors_sick++;

                seed = simpleRand(seed);
                if (neighbors_sick > 0 && (seed % 100) < (BETA * 100)) {
                    next_grid[i][j].sick = true;
                    next_grid[i][j].sick_days = 1;
                }
            }
        }
    }

    for (int i = start_row; i < end_row; ++i)
        for (int j = 0; j < COLS; ++j)
            grid[i][j] = next_grid[i][j];
}

// Store and output results at the end of the simulation
void outputResults(Person*** results) {
    std::ofstream file("flu_simulation_output.txt");
    for (int day = 0; day <= SIMULATION_DAYS; ++day) {
        file << "Day " << day << ":\n";
        for (int i = 0; i < ROWS; ++i) {
            for (int j = 0; j < COLS; ++j) {
                file << (results[day][i][j].sick ? "1 " : "0 ");
            }
            file << "\n";
        }
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        if (!readSettings("settings.txt")) {
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&ROWS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&COLS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ALPHA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BETA, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&OMEGA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&SIMULATION_DAYS, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int rows_per_proc = ROWS / size;
    int remainder_rows = ROWS % size;
    int start_row = rank * rows_per_proc + (rank < remainder_rows ? rank : remainder_rows);
    int end_row = start_row + rows_per_proc + (rank < remainder_rows ? 1 : 0);

    unsigned int seed = rank + 1;

    double start_time = MPI_Wtime();

    // Allocate grid and next_grid
    Person** grid = new Person * [ROWS];
    Person** next_grid = new Person * [ROWS];
    for (int i = 0; i < ROWS; ++i) {
        grid[i] = new Person[COLS];
        next_grid[i] = new Person[COLS];
    }

    if (rank == 0) {
        initializeGrid(grid, ROWS, seed);
    }

    // Broadcast initial grid to all processes
    for (int i = 0; i < ROWS; ++i) {
        MPI_Bcast(grid[i], COLS * sizeof(Person), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    Person*** results = nullptr;
    if (rank == 0) {
        results = new Person * *[SIMULATION_DAYS + 1];
        for (int day = 0; day <= SIMULATION_DAYS; ++day) {
            results[day] = new Person * [ROWS];
            for (int i = 0; i < ROWS; ++i) {
                results[day][i] = new Person[COLS];
            }
        }

        for (int i = 0; i < ROWS; ++i)
            for (int j = 0; j < COLS; ++j)
                results[0][i][j] = grid[i][j];
    }

    for (int day = 1; day <= SIMULATION_DAYS; ++day) {
        if (rank > 0) {
            MPI_Sendrecv(grid[start_row], COLS * sizeof(Person), MPI_BYTE, rank - 1, 0,
                grid[start_row - 1], COLS * sizeof(Person), MPI_BYTE, rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(grid[end_row - 1], COLS * sizeof(Person), MPI_BYTE, rank + 1, 0,
                grid[end_row], COLS * sizeof(Person), MPI_BYTE, rank + 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        simulateDay(grid, next_grid, start_row, end_row, seed);

        if (rank == 0) {
            for (int i = 0; i < ROWS; ++i)
                for (int j = 0; j < COLS; ++j)
                    results[day][i][j] = grid[i][j];
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Reduce timing to find maximum elapsed time
    double max_time;
    MPI_Reduce(&elapsed_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        outputResults(results);
        std::cout << "Maximum simulation time across all processes: " << max_time << " seconds.\n";

        for (int day = 0; day <= SIMULATION_DAYS; ++day) {
            for (int i = 0; i < ROWS; ++i) {
                delete[] results[day][i];
            }
            delete[] results[day];
        }
        delete[] results;
    }

    for (int i = 0; i < ROWS; ++i) {
        delete[] grid[i];
        delete[] next_grid[i];
    }
    delete[] grid;
    delete[] next_grid;

    MPI_Finalize();

    return 0;
}
