#include <lin_reg.h>

static void lin_reg_randomize_training_order(struct lin_reg* self) {
    for (size_t i = 0; i < self->num_sets; ++i) {
        const size_t r = rand() % self->num_sets;
        const size_t temp = self->train_order[i];
        self->train_order[i] = self->train_order[r];
        self->train_order[r] = temp;
    }
}

static void lin_reg_optimize(struct lin_reg* self, 
                             const double input, 
                             const double reference, 
                             const double learning_rate) {
    if (input != 0) {
        const double error = reference - lin_reg_predict(self, input);
        self->bias += error * learning_rate;
        self->weight += error * learning_rate * input;
    } else {
        self->bias = reference;
    }
}

static bool lin_reg_init_train_order_array(struct lin_reg* self) {
    self->train_order = (size_t*)malloc(sizeof(size_t) * self->num_sets);
    if (!self) return false;
    for (size_t i = 0; i < self->num_sets; ++i) {
        self->train_order[i] = i;
    }
    return true;
}

static void init_random_generator(void) {
    static bool random_generator_initialized = false;
    if (!random_generator_initialized) {
        srand(time(0));
        random_generator_initialized = true;
    }
}

bool lin_reg_init(struct lin_reg* self, 
                  const double* train_in, 
                  const double* train_out, 
                  const size_t num_sets) {
    self->train_order = 0;
    self->weight = 0;
    self->bias = 0;
    self->train_in = train_in;
    self->train_out = train_out;
    self->num_sets = num_sets;
    init_random_generator();
    return lin_reg_init_train_order_array(self);
}

void lin_reg_clear(struct lin_reg* self) {
    free(self->train_order);
    self->train_in = 0;
    self->train_out = 0;
    self->train_order = 0;
    self->num_sets = 0;
    self->bias = 0;
    self->weight = 0;
}

void lin_reg_train(struct lin_reg* self, 
                   const size_t num_epochs, 
                   const double learning_rate) {
    for (size_t i = 0; i < num_epochs; ++i) {
        lin_reg_randomize_training_order(self);
        for (size_t j = 0; j < self->num_sets; ++j) {
            const size_t k = self->train_order[j];
            lin_reg_optimize(self, self->train_in[k], self->train_out[k], learning_rate);
        }
    }
}