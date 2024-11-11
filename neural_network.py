import sqlite3 as sqlite
import numpy as np
from numpy import linalg as LA
import os
from typing import Dict


class Neural_Network(object):
    def __init__(self, topology: Dict[str, int], db_name='nn.db', delete_old_db=False, reg_lambda=0.0001):
        # Hyperparameters
        self._number_of_hidden_layers = topology['number_of_hidden_layers']
        self._input_layer_size = topology['input_layer_size']
        self._hidden_layer_size = topology['hidden_layer_size']
        self._output_layer_size = topology['output_layer_size']
        self._LAMBDA = reg_lambda

        if delete_old_db:
            self._delete_database(db_name=db_name)

        self._con = sqlite.connect(db_name)
        self._setup_network()

    ################################################################################################
    # PERSISTENCE
    ################################################################################################

    def __del__(self):
        self._con.close()

    def _delete_database(self, db_name: str):
        if os.path.exists(db_name):
            os.remove(db_name)

    def _setup_network(self):
        self._TABLE_STANDARDIZATION_PARAMETERS = 'standardization'
        self._TABLE_NAME_PREFIX = 'layer_'
        self._TABLE_NAME_FOR_LAYER = {}
        self._TABLE_NAME_WEIGHT_FOR_LAYERS = {}
        self._TABLE_NAME_BIAS_FOR_LAYERS = {}
        self._WEIGHTS_FOR_LAYER = {}
        self._BIAS_FOR_LAYER = {}
        self._OUTPUT_FOR_LAYER = {}

        # Input Nodes
        self._TABLE_NAME_FOR_LAYER[0] = f'{self._TABLE_NAME_PREFIX}input'

        # Output Nodes
        self._TABLE_NAME_FOR_LAYER[self._number_of_hidden_layers+1] = f'{self._TABLE_NAME_PREFIX}output'

        # Hidden Layers
        for l in range(1, self._number_of_hidden_layers+1):
            self._TABLE_NAME_FOR_LAYER[l] = f'{self._TABLE_NAME_PREFIX}hidden_{l}'

        # Weight Tables
        for l in sorted(self._TABLE_NAME_FOR_LAYER.keys())[:-1]:
            self._TABLE_NAME_WEIGHT_FOR_LAYERS[l] = f'weights_{self._TABLE_NAME_FOR_LAYER[l]}_to_{self._TABLE_NAME_FOR_LAYER[l+1]}'
            self._TABLE_NAME_BIAS_FOR_LAYERS[l] = f'bias_to_{self._TABLE_NAME_FOR_LAYER[l+1]}'

        self._create_tables()

        for from_layer in sorted(self._TABLE_NAME_FOR_LAYER.keys())[:-1]:
            to_layer = from_layer + 1
            from_ids = self._fetch_all_ids_from_table(table=self._TABLE_NAME_FOR_LAYER[from_layer])
            to_ids = self._fetch_all_ids_from_table(table=self._TABLE_NAME_FOR_LAYER[to_layer])

            # Load weights from db
            self._WEIGHTS_FOR_LAYER[from_layer] = np.array([[self._fetch_weight_from_db(r, c, from_layer) for c in to_ids] for r in from_ids])
            self._BIAS_FOR_LAYER[from_layer] = np.array([[self._fetch_bias_from_db(c, from_layer) for c in to_ids]])

    def _create_tables(self):
        self._TABLE_COLUMN_MEAN_PREFIX = 'mean_'
        self._TABLE_COLUMN_STD_PREFIX = 'std_'
        self._TABLE_COLUMN_ID = 'id'
        self._TABLE_COLUMN_FROM_ID = 'from_id'
        self._TABLE_COLUMN_TO_ID = 'to_id'
        self._TABLE_COLUMN_WEIGHT = 'weight'
        self._TABLE_COLUMN_TYPE = 'type'

        # Mean and Standard deviation table
        columns = "("
        columns += f"{self._TABLE_COLUMN_ID} INTEGER PRIMARY KEY,\n"
        separator = ""
        for i in range(self._input_layer_size):
            columns += f"{separator}{self._TABLE_COLUMN_MEAN_PREFIX}{i} REAL, \n{self._TABLE_COLUMN_STD_PREFIX}{i} REAL"
            separator = ",\n"
        columns += ")"
        self._con.execute(f"CREATE TABLE IF NOT EXISTS {self._TABLE_STANDARDIZATION_PARAMETERS} {columns}")
        self._con.commit()

        for layer in self._TABLE_NAME_FOR_LAYER:
            table = self._TABLE_NAME_FOR_LAYER[layer]
            self._con.execute(
                f'''CREATE TABLE IF NOT EXISTS {table}
                (
                    {self._TABLE_COLUMN_ID} INTEGER PRIMARY KEY
                )'''
            )
            self._con.commit()

        self._generate_and_persist_layer_nodes_to_db()

        # Weight Tables
        for from_layer in sorted(self._TABLE_NAME_FOR_LAYER.keys())[:-1]:
            to_layer = from_layer + 1
            from_table = self._TABLE_NAME_FOR_LAYER[from_layer]
            to_table = self._TABLE_NAME_FOR_LAYER[to_layer]
            weight_table = self._TABLE_NAME_WEIGHT_FOR_LAYERS[from_layer]
            self._con.execute(
                f'''CREATE TABLE IF NOT EXISTS {weight_table}
                (
                    {self._TABLE_COLUMN_ID} INTEGER PRIMARY KEY,
                    {self._TABLE_COLUMN_FROM_ID} INTEGER,
                    {self._TABLE_COLUMN_TO_ID} INTEGER,
                    {self._TABLE_COLUMN_WEIGHT} REAL,
                    FOREIGN KEY({self._TABLE_COLUMN_FROM_ID}) REFERENCES {from_table}({self._TABLE_COLUMN_ID}),
                    FOREIGN KEY({self._TABLE_COLUMN_TO_ID}) REFERENCES {to_table}({self._TABLE_COLUMN_ID}),
                    UNIQUE({self._TABLE_COLUMN_FROM_ID}, {self._TABLE_COLUMN_TO_ID})
                )'''
            )
            self._con.commit()

        for from_layer in sorted(self._TABLE_NAME_FOR_LAYER.keys())[:-1]:
            to_layer = from_layer + 1
            bias_table = self._TABLE_NAME_BIAS_FOR_LAYERS[from_layer]
            to_table = self._TABLE_NAME_FOR_LAYER[to_layer]
            self._con.execute(
                f'''CREATE TABLE IF NOT EXISTS {bias_table}
                (
                    {self._TABLE_COLUMN_ID} INTEGER PRIMARY KEY,
                    {self._TABLE_COLUMN_TO_ID} INTEGER,
                    {self._TABLE_COLUMN_WEIGHT} REAL,
                    FOREIGN KEY({self._TABLE_COLUMN_TO_ID}) REFERENCES {to_table}({self._TABLE_COLUMN_ID}),
                    UNIQUE({self._TABLE_COLUMN_TO_ID})
                )'''
            )
            self._con.commit()

        self._generate_and_persist_synapses_to_db()

    def _fetch_weight_from_db(self, from_id, to_id, from_layer):
        table = self._TABLE_NAME_WEIGHT_FOR_LAYERS[from_layer]

        res = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_WEIGHT}
            FROM
                {table}
            WHERE
                {self._TABLE_COLUMN_FROM_ID}={from_id}
                AND
                {self._TABLE_COLUMN_TO_ID}={to_id}
            '''
        ).fetchone()

        return res[0]

    def _fetch_bias_from_db(self, to_id, from_layer):
        table = self._TABLE_NAME_BIAS_FOR_LAYERS[from_layer]

        res = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_WEIGHT}
            FROM
                {table}
            WHERE
                {self._TABLE_COLUMN_TO_ID}={to_id}
            '''
        ).fetchone()

        return res[0]

    def _fetch_mean_std_from_db(self):
        table = self._TABLE_STANDARDIZATION_PARAMETERS

        columns = ""
        separator = ""
        for i in range(self._input_layer_size):
            columns += f"{separator}{self._TABLE_COLUMN_MEAN_PREFIX}{i}, {self._TABLE_COLUMN_STD_PREFIX}{i}"
            separator = ", "

        res = self._con.execute(
            f'''
            SELECT
                {columns}
            FROM
                {table}
            '''
        ).fetchone()

        means = []
        stds = []

        if res:
            for i in range(0, len(res), 2):
                means.append(res[i])
                stds.append(res[i+1])

        mean = np.array((means), dtype=float)
        std = np.array((stds), dtype=float)
        return mean, std

    def _persist_mean_std_to_db(self, mean, std):
        table = self._TABLE_STANDARDIZATION_PARAMETERS

        res = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_ID}
            FROM
                {table}
            '''
        ).fetchone()

        if res:
            row_id = res[0]

            columns_update = ""
            separator = ""
            for i in range(self._input_layer_size):
                columns_update += f"{separator}{self._TABLE_COLUMN_MEAN_PREFIX}{i}={mean[i]}, {self._TABLE_COLUMN_STD_PREFIX}{i}={std[i]}"
                separator = ", "

            self._con.execute(
                f'''
                UPDATE {table}
                SET
                    {columns_update}
                WHERE
                    {self._TABLE_COLUMN_ID}={row_id}
                '''
            )
        else:
            columns_insert = "("
            separator = ""
            for i in range(self._input_layer_size):
                columns_insert += f"{separator}{self._TABLE_COLUMN_MEAN_PREFIX}{i}, {self._TABLE_COLUMN_STD_PREFIX}{i}"
                separator = ", "
            columns_insert += ")"

            values_insert = "("
            separator = ""
            for i in range(self._input_layer_size):
                values_insert += f"{separator}{mean[i]}, {std[i]}"
                separator = ", "
            values_insert += ")"

            self._con.execute(
                f'''
                INSERT INTO {table}
                    {columns_insert}
                VALUES
                    {values_insert}
                '''
            )
        self._con.commit()

    def _persist_weight_to_db(self, from_id, to_id, from_layer, weight):
        table = self._TABLE_NAME_WEIGHT_FOR_LAYERS[from_layer]

        res = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_ID}
            FROM
                {table}
            WHERE
                {self._TABLE_COLUMN_FROM_ID}={from_id}
                AND
                {self._TABLE_COLUMN_TO_ID}={to_id}
            '''
        ).fetchone()

        row_id = res[0]

        self._con.execute(
            f'''
            UPDATE {table}
            SET
                {self._TABLE_COLUMN_WEIGHT}={weight}
            WHERE
                {self._TABLE_COLUMN_ID}={row_id}
            '''
        )

        self._con.commit()

    def _persist_bias_to_db(self, to_id, from_layer, weight):
        table = self._TABLE_NAME_BIAS_FOR_LAYERS[from_layer]

        res = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_ID}
            FROM
                {table}
            WHERE
                {self._TABLE_COLUMN_TO_ID}={to_id}
            '''
        ).fetchone()

        row_id = res[0]

        self._con.execute(
            f'''
            UPDATE {table}
            SET
                {self._TABLE_COLUMN_WEIGHT}={weight}
            WHERE
                {self._TABLE_COLUMN_ID}={row_id}
            '''
        )

        self._con.commit()

    def _fetch_all_ids_from_table(self, table: str):
        result = set()

        cur = self._con.execute(
            f'''
            SELECT
                {self._TABLE_COLUMN_ID}
            FROM
                {table}
            '''
        )
        for row in cur:
            result.add(row[0])

        return sorted(list(result))

    def _generate_and_persist_layer_nodes_to_db(self):
        for layer in sorted(self._TABLE_NAME_FOR_LAYER.keys()):
            table = self._TABLE_NAME_FOR_LAYER[layer]
            ids = self._fetch_all_ids_from_table(table=table)
            if len(ids) == 0:
                if layer == 0:
                    number_of_nodes = self._input_layer_size
                elif layer == self._number_of_hidden_layers+1:
                    number_of_nodes = self._output_layer_size
                else:
                    number_of_nodes = self._hidden_layer_size

                for _ in range(number_of_nodes):
                    cur = self._con.execute(
                        f"""
                        INSERT INTO {table} DEFAULT VALUES;
                        """
                    )
                self._con.commit()

    def _generate_and_persist_synapses_to_db(self):
        for from_layer in sorted(self._TABLE_NAME_FOR_LAYER.keys())[:-1]:
            to_layer = from_layer + 1
            from_table = self._TABLE_NAME_FOR_LAYER[from_layer]
            to_table = self._TABLE_NAME_FOR_LAYER[to_layer]
            weight_table = self._TABLE_NAME_WEIGHT_FOR_LAYERS[from_layer]
            bias_table = self._TABLE_NAME_BIAS_FOR_LAYERS[from_layer]

            from_ids = self._fetch_all_ids_from_table(table=from_table)
            to_ids = self._fetch_all_ids_from_table(table=to_table)
            initial_weight = 1.0 / self._hidden_layer_size if from_layer == 0 else 1

            for from_id in from_ids:
                for to_id in to_ids:
                    cur = self._con.execute(
                        f"""
                        INSERT OR IGNORE INTO {weight_table}
                        (
                            {self._TABLE_COLUMN_FROM_ID},
                            {self._TABLE_COLUMN_TO_ID},
                            {self._TABLE_COLUMN_WEIGHT}
                        )
                        VALUES
                        (
                            {from_id},
                            {to_id},
                            {initial_weight}
                        )
                        """
                    )
                    self._con.commit()

            for to_id in to_ids:
                cur = self._con.execute(
                    f"""
                    INSERT OR IGNORE INTO {bias_table}
                    (
                        {self._TABLE_COLUMN_TO_ID},
                        {self._TABLE_COLUMN_WEIGHT}
                    )
                    VALUES
                    (
                        {to_id},
                        {initial_weight}
                    )
                    """
                )
                self._con.commit()

    def _update_db_weights_and_bias(self):
        for layer in sorted(self._WEIGHTS_FOR_LAYER.keys()):
            W = self._WEIGHTS_FOR_LAYER[layer]
            rows, columns = W.shape
            for row in range(rows):
                for col in range(columns):
                    self._persist_weight_to_db(from_id=row+1, to_id=col+1, from_layer=layer, weight=W[row][col])

        for layer in sorted(self._BIAS_FOR_LAYER.keys()):
            b = self._BIAS_FOR_LAYER[layer]
            rows, columns = b.shape
            for row in range(rows):
                for col in range(columns):
                    self._persist_bias_to_db(to_id=col+1, from_layer=layer, weight=b[row][col])

    ################################################################################################
    # LEARNING & TRAINING
    ################################################################################################

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _dsigmoid(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def _derivatives_with_respect_to_weights(self, X, y):
        N = X.shape[0]
        dJdW_for_layer = {}
        deltas_for_layer = {}

        # Compute derivative with respect to W and W2 for a given X and y:
        yHat = self.forward(X)

        # Output Layer
        W = self._WEIGHTS_FOR_LAYER[self._number_of_hidden_layers]
        bias = self._BIAS_FOR_LAYER[self._number_of_hidden_layers]
        net = np.dot(self._OUTPUT_FOR_LAYER[self._number_of_hidden_layers], W) + bias
        # Compute the average gradient by dividing by sample size X.shape[0]. Also add gradient of regularization term.
        deltas_for_layer[self._number_of_hidden_layers] = np.multiply(-(y - yHat), self._dsigmoid(net))
        dJdW_for_layer[self._number_of_hidden_layers] = np.dot(self._OUTPUT_FOR_LAYER[self._number_of_hidden_layers].T, deltas_for_layer[self._number_of_hidden_layers])/N + self._LAMBDA*W

        # Hidden Layers
        for layer in sorted(self._TABLE_NAME_FOR_LAYER.keys(), reverse=True)[2:]:
            W1 = self._WEIGHTS_FOR_LAYER[layer]
            W2 = self._WEIGHTS_FOR_LAYER[layer + 1]
            bias = self._BIAS_FOR_LAYER[layer]
            net = np.dot(self._OUTPUT_FOR_LAYER[layer], W1) + bias
            delta = deltas_for_layer[layer + 1]
            deltas_for_layer[layer] = np.dot(delta, W2.T)*self._dsigmoid(net)
            dJdW_for_layer[layer] = np.dot(self._OUTPUT_FOR_LAYER[layer].T, deltas_for_layer[layer])/N + self._LAMBDA*W1

        return dJdW_for_layer, deltas_for_layer

    def cost_function(self, X, y):
        '''
        Return cost given input X and known output y.

        cost = 0.5 * sum((y - yHat)^2)/N + r

        with:
            N: number of samples
            r: regularization
            yHat: predicted output
        '''
        N = X.shape[0]

        yHat = self.forward(X)

        W = self._WEIGHTS_FOR_LAYER[0]
        reg = np.sum(W**2)
        for layer in sorted(self._WEIGHTS_FOR_LAYER.keys())[1:]:
            W = self._WEIGHTS_FOR_LAYER[layer]
            reg = reg + np.sum(W**2)

        # We don't want cost to increase with the number of examples,
        # so normalize by dividing the error term by number of examples(X.shape[0])
        J = 0.5*sum((y-yHat)**2)/N + (self._LAMBDA/2)*reg
        return J

    def _standardize(self, X, y, persist=True):
        '''
        Standardizes X and y vectors and persists the input vector (X)
        mean and standard deviation to the database for future use.

        Procedure:
        z=(X-μ)/std
        1. Find mean (μ) and standard deviation for each feature or output vector.
        2. Subtract mean (μ) from the samples (X) or output y.
        3. Divide with standard deviation.

        Returns standardized X and y vectors.
        '''
        mean, std = self._fetch_mean_std_from_db()
        if mean.size == 0 and std.size == 0:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            if persist:
                self._persist_mean_std_to_db(mean, std)

        X_std = (X-mean)/std
        y_std = (y - np.mean(y, axis=0))/np.std(y, axis=0)

        return X_std, y_std

    def transform(self, X, y):
        '''
        Returns min-max normalized X and y vectors.
        '''
        X_scaled = (X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))
        y_scaled = (y - np.min(y, axis=0))/(np.max(y, axis=0) - np.min(y, axis=0))

        return X_scaled, y_scaled

    # Gradient Descent
    def train_using_gradient_descent(self, X, y, learning_rate=0.5):
        '''
        Train network using gradient descent.
        '''
        N = X.shape[0]
        dJdW_for_layer, deltas_for_layer = self._derivatives_with_respect_to_weights(X, y)
        for layer in sorted(self._WEIGHTS_FOR_LAYER.keys()):
            W = self._WEIGHTS_FOR_LAYER[layer]
            self._WEIGHTS_FOR_LAYER[layer] -= learning_rate * dJdW_for_layer[layer]
            self._BIAS_FOR_LAYER[layer] -= learning_rate * np.sum(deltas_for_layer[layer], axis=0, keepdims=True)/N
        self._update_db_weights_and_bias()

    def forward(self, X):
        '''
        Propogate inputs though network.
        '''
        self._OUTPUT_FOR_LAYER[0] = X
        for layer in sorted(self._TABLE_NAME_FOR_LAYER.keys())[1:]:
            W = self._WEIGHTS_FOR_LAYER[layer - 1]
            bias = self._BIAS_FOR_LAYER[layer - 1]
            net = np.dot(self._OUTPUT_FOR_LAYER[layer - 1], W) + bias
            self._OUTPUT_FOR_LAYER[layer] = self._sigmoid(net)

        yHat = self._OUTPUT_FOR_LAYER[self._number_of_hidden_layers+1]
        return yHat


def main():
    topology = {
        'input_layer_size': 1,
        'number_of_hidden_layers': 1,
        'hidden_layer_size': 3,
        'output_layer_size': 1
    }
    net = Neural_Network(topology=topology, delete_old_db=True)


if __name__ == "__main__":
    main()
