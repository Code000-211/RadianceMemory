# tools/ -- rad-info, rad-convert, rad-kbench, rad-tune, and the smaller checkers below.
#
# rad_add_tool tools link rad_core: rad-convert, rad-kbench and rad-tune are the engine's startup
# with a different last step, loading the same plugins in the same hierarchy and running the same
# declare phase (spec §4.4, §15, §17), so they resolve kernels exactly as the engine does.
#
# iface.cpp is the seam onto core/plugin and core/build and the flag parsing every tool shares,
# opshapes.cpp the operand sizing shared by rad-kbench and rad-tune, kfixture.cpp the recorded
# kernel fixture and its input draw, and kbench_core.cpp rad-kbench's rules (comparison,
# tolerances, keys, the reference cache). They are compiled into each tool; a library target is not
# worth it for four files.

set(RAD_TOOL_SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/iface.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/opshapes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kfixture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kbench_core.cpp)

function(rad_add_tool name src)
  add_executable(${name} ${src} ${RAD_TOOL_SHARED})
  # WHOLE_ARCHIVE: a host that dlopens a plugin must present the whole plugin-facing ABI, declare
  # and run phase alike. A plain archive link drops objects the tool does not reference (e.g.
  # rad-convert never calls rad_issue), and the architecture plugin then fails to load.
  target_link_libraries(${name} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/abi)
  # No home definition here: rad_abi carries RAD_DEFAULT_HOME to the engine and tools alike, and
  # iface.cpp's rad_home() reads it, so there is one default home.
  #
  # ENABLE_EXPORTS is required: the host implements the declare-phase C ABI (rad_decl_weight,
  # rad_decl_op, rad_declare_logits, ...) and an architecture plugin binds to it at dlopen. Kernel
  # plugins only export, so rad-schemas and r4d_selftest do not need it.
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    ENABLE_EXPORTS ON)
endfunction()

rad_add_tool(rad-info     rad_info.cpp)

# The tokeniser's independent reference encoder (tokref.h), which core/text's pre-tokeniser is
# checked against. A library because tests/tokenizer_test uses it too; core/ does not link it.
add_library(rad_tokref STATIC ${CMAKE_CURRENT_SOURCE_DIR}/tokref.cpp)
target_link_libraries(rad_tokref PUBLIC rad_core)
target_include_directories(rad_tokref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/core)

# rad-tokcheck needs neither plugins nor the declare phase, so it links the core plainly.
add_executable(rad-tokcheck rad_tokcheck.cpp ${CMAKE_CURRENT_SOURCE_DIR}/iface.cpp)
target_link_libraries(rad-tokcheck PRIVATE rad_tokref rad_core)
set_target_properties(rad-tokcheck PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# rad-autoquant runs the engine as a child for every measurement and loads no plugin itself, so like
# rad-tokcheck it links the core plainly: the recipe parser and the solver (core/kv_autoquant.h).
add_executable(rad-autoquant rad_autoquant.cpp)
target_link_libraries(rad-autoquant PRIVATE rad_core)
set_target_properties(rad-autoquant PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# rad-schemas links no core: it dlopens plugins and reads their static schema tables, so it works on
# a plugin the engine cannot load.
add_executable(rad-schemas rad_schemas.cpp)
target_link_libraries(rad-schemas PRIVATE rad_abi ${CMAKE_DL_LIBS})
set_target_properties(rad-schemas PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
rad_add_tool(rad-convert  rad_convert.cpp)
# Optional OpenMP across weights and within a large one.
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(rad-convert PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "rad-convert: no OpenMP -- conversion will run single-threaded")
endif()
rad_add_tool(rad-kbench    rad_kbench.cpp)
# The timing pass (--bench): the only part of rad-kbench that needs the device layer's internals.
target_sources(rad-kbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kbench_timer.cpp)
rad_add_tool(rad-tune     rad_tune.cpp)

# GBNF mask engine acceptance run: tests/gbnf_test.cpp's differential and latency suite against a
# real tokenizer.json and chat template. It loads no plugin, and is not installed because it
# compiles in test code (tests/gbnf_oracle.cpp, upstream's candidate walk) as its oracle.
add_executable(rad-gbnf-check rad_gbnf_check.cpp ${CMAKE_SOURCE_DIR}/tests/gbnf_oracle.cpp
                              ${CMAKE_CURRENT_SOURCE_DIR}/iface.cpp)
target_link_libraries(rad-gbnf-check PRIVATE rad_core)
target_include_directories(rad-gbnf-check PRIVATE
  ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(rad-gbnf-check PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# rad-schemas matters most in an installed prefix: it needs only the four exports every kernel
# plugin has, so it can debug an out-of-tree plugin the engine cannot load.
install(TARGETS rad-info rad-convert rad-kbench rad-tune rad-schemas rad-tokcheck rad-autoquant
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# rad-chatparse: the reply parser's acceptance run against a checkpoint's real tokenizer and chat
# template, compared with the legacy parser in tests/legacy/. It shares that parser and the
# differential harness with tests/chatparse_test, so it is built only with the tests.
if(RAD_BUILD_TESTS)
  # Its --request mode loads the architecture plugin for the model's declared reply format, so it
  # is a plugin host like rad_add_tool's tools: the whole of rad_core, with its symbols exported.
  # The legacy parser's sources are compiled in rather than linked as rad_legacy_toolparse, which
  # links rad_core without that feature.
  add_executable(rad-chatparse rad_chatparse.cpp chatparse_serve.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/iface.cpp
                               ${CMAKE_SOURCE_DIR}/tests/legacy/toolparse_legacy.cpp
                               ${CMAKE_SOURCE_DIR}/tests/legacy/chatdiff.cpp)
  target_link_libraries(rad-chatparse PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,rad_core>")
  target_include_directories(rad-chatparse PRIVATE ${CMAKE_SOURCE_DIR}/core ${CMAKE_SOURCE_DIR}/abi
                                                   ${CMAKE_CURRENT_SOURCE_DIR}
                                                   ${CMAKE_SOURCE_DIR}/tests/legacy)
  set_target_properties(rad-chatparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                                                 ENABLE_EXPORTS ON)
endif()
