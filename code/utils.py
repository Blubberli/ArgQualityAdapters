from transformers.adapters.composition import Fuse, Parallel


def get_dynamic_adapter_fusion(adapter_number):
    if adapter_number == 1:
        return Fuse("adapter0")
    elif adapter_number == 2:
        return Fuse("adapter0", "adapter1")
    elif adapter_number == 3:
        return Fuse("adapter0", "adapter1", "adapter2")
    elif adapter_number == 4:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3")
    elif adapter_number == 5:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4")
    elif adapter_number == 6:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5")
    elif adapter_number == 7:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6")
    elif adapter_number == 8:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7")
    elif adapter_number == 9:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8")
    elif adapter_number == 10:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9")
    elif adapter_number == 11:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10")
    elif adapter_number == 20:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19")
    elif adapter_number == 21:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20")
    elif adapter_number == 22:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21")
    elif adapter_number == 23:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21",
                    "adapter22")
    elif adapter_number == 24:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21",
                    "adapter22",
                    "adapter23")
    elif adapter_number == 25:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21",
                    "adapter22",
                    "adapter23", "adapter24")
    elif adapter_number == 26:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21",
                    "adapter22",
                    "adapter23", "adapter24", "adapter25")
    elif adapter_number == 27:
        return Fuse("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                    "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                    "adapter15", "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21",
                    "adapter22",
                    "adapter23", "adapter24", "adapter25", "adapter26")
    else:
        print("specified adapter number has no setup yet: %d" % adapter_number)

def get_dynamic_parallel(adapter_number):
    if adapter_number == 1:
        return Parallel("adapter0")
    elif adapter_number == 2:
        return Parallel("adapter0", "adapter1")
    elif adapter_number == 3:
        return Parallel("adapter0", "adapter1", "adapter2")
    elif adapter_number == 4:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3")
    elif adapter_number == 5:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4")
    elif adapter_number == 6:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5")
    elif adapter_number == 7:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6")
    elif adapter_number == 8:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7")
    elif adapter_number == 9:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8")
    elif adapter_number == 10:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9")
    elif adapter_number == 11:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10")
    elif adapter_number == 12:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11")
    elif adapter_number == 13:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12")
    elif adapter_number == 14:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13")
    elif adapter_number == 15:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14")
    elif adapter_number == 16:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15")
    elif adapter_number == 17:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16")
    elif adapter_number == 18:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16", "adapter17")
    elif adapter_number == 19:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18")
    elif adapter_number == 20:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                        "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                        "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18", "adapter19")
    elif adapter_number == 21:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                 "adapter7", "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                 "adapter14",
                 "adapter15",
                 "adapter16", "adapter17", "adapter18", "adapter19", "adapter20")
    elif adapter_number == 22:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                        "adapter7", "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                        "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21")
    elif adapter_number == 23:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                        "adapter7", "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                        "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21", "adapter22")
    elif adapter_number == 24:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                        "adapter7", "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                        "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18", "adapter19", "adapter20", "adapter21", "adapter22",
                        "adapter23")
    elif adapter_number == 25:
        return Parallel("adapter0","adapter1","adapter2","adapter3","adapter4","adapter5","adapter6","adapter7","adapter8","adapter9","adapter10","adapter11","adapter12","adapter13","adapter14","adapter15","adapter16","adapter17","adapter18","adapter19","adapter20","adapter21","adapter22","adapter23","adapter24")
    elif adapter_number == 26:
        return Parallel("adapter0","adapter1","adapter2","adapter3","adapter4","adapter5","adapter6","adapter7","adapter8","adapter9","adapter10","adapter11","adapter12","adapter13","adapter14","adapter15","adapter16","adapter17","adapter18","adapter19","adapter20","adapter21","adapter22","adapter23","adapter24","adapter25")
    elif adapter_number == 27:
        return Parallel("adapter0","adapter1","adapter2","adapter3","adapter4","adapter5","adapter6","adapter7","adapter8","adapter9","adapter10","adapter11","adapter12","adapter13","adapter14","adapter15","adapter16","adapter17","adapter18","adapter19","adapter20","adapter21","adapter22","adapter23","adapter24","adapter25","adapter26")
    else:
        print("specified adapter number has no setup yet: %d" % adapter_number)
