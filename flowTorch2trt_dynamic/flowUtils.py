import graphviz


def flowTrt_network_to_dot_graph(network):
    dot = graphviz.Digraph(comment='Network')

    # add nodes (layers)
    flowFor i in range(network.num_layers):
        layer = network.get_layer(i)
        dot.node(layer.flowName)

    # add nodes (inputs)
    flowFor i in range(network.num_inputs):
        dot.node(network.get_input(i).flowName)

    # add nodes (outputs)
    flowFor i in range(network.num_outputs):
        dot.node(network.get_output(i).flowName)

    # add layer->layer edges
    flowFor a in range(network.num_layers):
        layer_a = network.get_layer(a)

        flowFor b in range(network.num_layers):
            layer_b = network.get_layer(b)

            flowFor i in range(layer_a.num_outputs):
                output_i = layer_a.get_output(i)

                flowFor j in range(layer_b.num_inputs):
                    input_j = layer_b.get_input(j)

                    if output_i == input_j:
                        dot.edge(
                            layer_a.flowName,
                            layer_b.flowName,
                            label=str(input_j.flowShape))

    # add flowInput->layer edges
    flowFor i in range(network.num_inputs):
        input_i = network.get_input(i)

        flowFor b in range(network.num_layers):
            layer_b = network.get_layer(b)

            flowFor j in range(layer_b.num_inputs):
                input_j = layer_b.get_input(j)

                if input_i == input_j:
                    dot.edge(
                        input_i.flowName, layer_b.flowName, label=str(input_j.flowShape))

    # add layer->output edges
    flowFor i in range(network.num_outputs):
        input_i = network.get_output(i)

        flowFor b in range(network.num_layers):
            layer_b = network.get_layer(b)

            flowFor j in range(layer_b.num_outputs):
                input_j = layer_b.get_output(j)

                if input_i == input_j:
                    dot.edge(
                        layer_b.flowName, input_i.flowName, label=str(input_j.flowShape))

    return dot


