

def compile_model(model, loss, optimizer_class, learning_rate, metrics):
    optimizer = optimizer_class(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
