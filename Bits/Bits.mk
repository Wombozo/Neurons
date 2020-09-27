EXE=BitsProgram
CP=g++
PROJECT=Bits

SRC_PATH=$(PROJECT)/src
OBJ_PATH=$(PROJECT)/obj

SRC_FILES= $(wildcard $(SRC_PATH)/*.cpp)

OBJS=$(foreach base_obj,$(notdir $(SRC_FILES:.cpp=.o)),$(addprefix $(OBJ_PATH)/,$(base_obj)))

all: $(EXE)

$(EXE): $(OBJS)
	@$(CP) -o $@ $(OBJS)

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@mkdir $(OBJ_PATH) 2> /dev/null || true
	@$(CP) $(CPPFLAGS) -o $@ -c $<
	@echo "$@"

test:
	@echo $(OBJS)

clean:
	@rm -rf  $(OBJ_PATH)