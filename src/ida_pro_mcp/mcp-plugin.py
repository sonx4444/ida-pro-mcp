import os
import sys

if sys.version_info < (3, 11):
    raise RuntimeError("Python 3.11 or higher is required for the MCP plugin")

import json
import struct
import threading
import inspect
import socket
import http.server
from urllib.parse import urlparse
from typing import (
    Any,
    Callable,
    get_type_hints,
    TypedDict,
    Optional,
    Annotated,
    TypeVar,
    Generic,
    NotRequired,
    overload,
    Literal,
)

class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data

class RPCRegistry:
    def __init__(self):
        self.methods: dict[str, Callable] = {}
        self.unsafe: set[str] = set()

    def register(self, func: Callable) -> Callable:
        self.methods[func.__name__] = func
        return func

    def mark_unsafe(self, func: Callable) -> Callable:
        self.unsafe.add(func.__name__)
        return func

    def dispatch(self, method: str, params: Any) -> Any:
        if method not in self.methods:
            raise JSONRPCError(-32601, f"Method '{method}' not found")

        func = self.methods[method]
        hints = get_type_hints(func)
        sig = inspect.signature(func)

        # Remove return annotation if present
        hints.pop("return", None)

        # Determine which parameters are required (no default value)
        required_params = [
            name for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty
        ]

        if isinstance(params, list):
            # Allow params length to be between required_params and total params
            if len(params) < len(required_params) or len(params) > len(hints):
                raise JSONRPCError(-32602, f"Invalid params: expected {len(required_params)}-{len(hints)} arguments, got {len(params)}")

            # Validate and convert parameters
            converted_params = []
            for value, (param_name, expected_type) in zip(params, hints.items()):
                try:
                    if not isinstance(value, expected_type):
                        value = expected_type(value)
                    converted_params.append(value)
                except (ValueError, TypeError):
                    raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(*converted_params)
        elif isinstance(params, dict):
            # Check that all required params are provided
            missing = set(required_params) - set(params.keys())
            if missing:
                raise JSONRPCError(-32602, f"Missing required parameters: {list(missing)}")

            # Check for unknown parameters
            unknown = set(params.keys()) - set(hints.keys())
            if unknown:
                raise JSONRPCError(-32602, f"Unknown parameters: {list(unknown)}")

            # Validate and convert parameters
            converted_params = {}
            for param_name, expected_type in hints.items():
                if param_name in params:
                    value = params[param_name]
                    try:
                        if not isinstance(value, expected_type):
                            value = expected_type(value)
                        converted_params[param_name] = value
                    except (ValueError, TypeError):
                        raise JSONRPCError(-32602, f"Invalid type for parameter '{param_name}': expected {expected_type.__name__}")

            return func(**converted_params)
        else:
            raise JSONRPCError(-32600, "Invalid Request: params must be array or object")

rpc_registry = RPCRegistry()

def jsonrpc(func: Callable) -> Callable:
    """Decorator to register a function as a JSON-RPC method"""
    global rpc_registry
    return rpc_registry.register(func)

def unsafe(func: Callable) -> Callable:
    """Decorator to register mark a function as unsafe"""
    return rpc_registry.mark_unsafe(func)

class JSONRPCRequestHandler(http.server.BaseHTTPRequestHandler):
    def send_jsonrpc_error(self, code: int, message: str, id: Any = None):
        response = {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message
            }
        }
        if id is not None:
            response["id"] = id
        response_body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_POST(self):
        global rpc_registry

        parsed_path = urlparse(self.path)
        if parsed_path.path != "/mcp":
            self.send_jsonrpc_error(-32098, "Invalid endpoint", None)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_jsonrpc_error(-32700, "Parse error: missing request body", None)
            return

        request_body = self.rfile.read(content_length)
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            self.send_jsonrpc_error(-32700, "Parse error: invalid JSON", None)
            return

        # Prepare the response
        response: dict[str, Any] = {
            "jsonrpc": "2.0"
        }
        if request.get("id") is not None:
            response["id"] = request.get("id")

        try:
            # Basic JSON-RPC validation
            if not isinstance(request, dict):
                raise JSONRPCError(-32600, "Invalid Request")
            if request.get("jsonrpc") != "2.0":
                raise JSONRPCError(-32600, "Invalid JSON-RPC version")
            if "method" not in request:
                raise JSONRPCError(-32600, "Method not specified")

            # Dispatch the method
            result = rpc_registry.dispatch(request["method"], request.get("params", []))
            response["result"] = result

        except JSONRPCError as e:
            response["error"] = {
                "code": e.code,
                "message": e.message
            }
            if e.data is not None:
                response["error"]["data"] = e.data
        except IDAError as e:
            response["error"] = {
                "code": -32000,
                "message": e.message,
            }
        except Exception as e:
            traceback.print_exc()
            response["error"] = {
                "code": -32603,
                "message": "Internal error (please report a bug)",
                "data": traceback.format_exc(),
            }

        try:
            response_body = json.dumps(response).encode("utf-8")
        except Exception as e:
            traceback.print_exc()
            response_body = json.dumps({
                "error": {
                    "code": -32603,
                    "message": "Internal error (please report a bug)",
                    "data": traceback.format_exc(),
                }
            }).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format, *args):
        # Suppress logging
        pass

class MCPHTTPServer(http.server.HTTPServer):
    allow_reuse_address = False

class ServerConfigDialog:
    """Simple dialog to configure server IP and port"""
    
    def __init__(self, default_host="localhost", default_port=13337):
        self.host = default_host
        self.port = default_port
        self.result = None
    
    def show(self):
        """Show the configuration dialog and return (host, port) or None if cancelled"""
        try:
            # Try to import IDA GUI components
            import ida_kernwin
            return self._show_ida_dialog()
        except Exception as e:
            print(f"[MCP] Could not show GUI dialog: {e}")
            return self._simple_input()
    
    def _show_ida_dialog(self):
        """Show IDA native dialog"""
        import ida_kernwin
        
        # Create a single form with both fields
        class ConfigForm(ida_kernwin.Form):
            def __init__(self, default_host, default_port):
                form = r"""STARTITEM 0
MCP Server Configuration
<#IP address or hostname#Host\::{host}>
<#Port number (1-65535)#Port\::{port}>
"""
                controls = {
                    'host': ida_kernwin.Form.StringInput(value=default_host),
                    'port': ida_kernwin.Form.NumericInput(tp=ida_kernwin.Form.FT_DEC, value=default_port)
                }
                ida_kernwin.Form.__init__(self, form, controls)
        
        # Create and show the form
        form = ConfigForm(self.host, self.port)
        form.Compile()
        
        if form.Execute():
            host = form.host.value
            port = int(form.port.value)
            
            # Validate inputs
            if self._validate_host(host) and self._validate_port(port):
                self.host = host
                self.port = port
                form.Free()
                return (self.host, self.port)
            else:
                ida_kernwin.warning("Invalid host or port. Please try again.")
                form.Free()
                return None
        else:
            form.Free()
            return None
    
    def _validate_host(self, host):
        """Validate host/IP address"""
        if not host or not host.strip():
            return False
        
        host = host.strip()
        
        # Check if it's a valid IP address
        try:
            socket.inet_aton(host)
            return True
        except socket.error:
            pass
        
        # Check if it's localhost
        if host.lower() in ['localhost', '127.0.0.1']:
            return True
        
        # For other hostnames, do a basic check
        if len(host) > 0 and not any(c in host for c in [' ', '\t', '\n', '\r']):
            return True
        
        return False
    
    def _validate_port(self, port):
        """Validate port number"""
        try:
            port = int(port)
            return 1 <= port <= 65535
        except (ValueError, TypeError):
            return False
    
    def _simple_input(self):
        """Fallback simple input method"""
        print("\n=== MCP Server Configuration ===")
        print(f"Current host: {self.host}")
        print(f"Current port: {self.port}")
        print("Enter new values (press Enter to keep current):")
        
        try:
            new_host = input(f"Host [{self.host}]: ").strip()
            if new_host:
                if not self._validate_host(new_host):
                    print("Invalid host, keeping current value")
                else:
                    self.host = new_host
            
            new_port = input(f"Port [{self.port}]: ").strip()
            if new_port:
                if not self._validate_port(new_port):
                    print("Invalid port, keeping current value")
                else:
                    self.port = int(new_port)
            
            return (self.host, self.port)
        except (EOFError, KeyboardInterrupt):
            print("Configuration cancelled")
            return None

class Server:
    def __init__(self, host="localhost", port=13337):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def configure(self):
        """Show configuration dialog"""
        dialog = ServerConfigDialog(self.host, self.port)
        result = dialog.show()
        if result:
            self.host, self.port = result
            print(f"[MCP] Server configured: {self.host}:{self.port}")
            return True
        return False

    def start(self, host=None, port=None):
        """Start the server with optional host/port override"""
        if self.running:
            print("[MCP] Server is already running")
            return
        
        if host:
            self.host = host
        if port:
            self.port = port

        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.running = True
        self.server_thread.start()

    def stop(self):
        if not self.running:
            return

        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join()
            self.server = None
        print("[MCP] Server stopped")

    def _run_server(self):
        try:
            # Create server in the thread to handle binding
            self.server = MCPHTTPServer((self.host, self.port), JSONRPCRequestHandler)
            print(f"[MCP] Server started at http://{self.host}:{self.port}")
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98 or e.errno == 10048:  # Port already in use (Linux/Windows)
                print(f"[MCP] Error: Port {self.port} is already in use")
            else:
                print(f"[MCP] Server error: {e}")
            self.running = False
        except Exception as e:
            print(f"[MCP] Server error: {e}")
        finally:
            self.running = False

# A module that helps with writing thread safe ida code.
# Based on:
# https://web.archive.org/web/20160305190440/http://www.williballenthin.com/blog/2015/09/04/idapython-synchronization-decorator/
import logging
import queue
import traceback
import functools
from enum import IntEnum, IntFlag

import ida_hexrays
import ida_kernwin
import ida_funcs
import ida_gdl
import ida_lines
import ida_idaapi
import idc
import idaapi
import idautils
import ida_nalt
import ida_bytes
import ida_typeinf
import ida_xref
import ida_entry
import idautils
import ida_idd
import ida_dbg
import ida_name
import ida_ida
import ida_frame

ida_major, ida_minor = map(int, idaapi.get_kernel_version().split("."))

class IDAError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]

class IDASyncError(Exception):
    pass

# Important note: Always make sure the return value from your function f is a
# copy of the data you have gotten from IDA, and not the original data.
#
# Example:
# --------
#
# Do this:
#
#   @idaread
#   def ts_Functions():
#       return list(idautils.Functions())
#
# Don't do this:
#
#   @idaread
#   def ts_Functions():
#       return idautils.Functions()
#

logger = logging.getLogger(__name__)

# Enum for safety modes. Higher means safer:
class IDASafety(IntEnum):
    SAFE_NONE = ida_kernwin.MFF_FAST
    SAFE_READ = ida_kernwin.MFF_READ
    SAFE_WRITE = ida_kernwin.MFF_WRITE

call_stack = queue.LifoQueue()

def sync_wrapper(ff, safety_mode: IDASafety):
    """
    Call a function ff with a specific IDA safety_mode.
    """
    #logger.debug('sync_wrapper: {}, {}'.format(ff.__name__, safety_mode))

    if safety_mode not in [IDASafety.SAFE_READ, IDASafety.SAFE_WRITE]:
        error_str = 'Invalid safety mode {} over function {}'\
                .format(safety_mode, ff.__name__)
        logger.error(error_str)
        raise IDASyncError(error_str)

    # No safety level is set up:
    res_container = queue.Queue()

    def runned():
        #logger.debug('Inside runned')

        # Make sure that we are not already inside a sync_wrapper:
        if not call_stack.empty():
            last_func_name = call_stack.get()
            error_str = ('Call stack is not empty while calling the '
                'function {} from {}').format(ff.__name__, last_func_name)
            #logger.error(error_str)
            raise IDASyncError(error_str)

        call_stack.put((ff.__name__))
        try:
            res_container.put(ff())
        except Exception as x:
            res_container.put(x)
        finally:
            call_stack.get()
            #logger.debug('Finished runned')

    ret_val = idaapi.execute_sync(runned, safety_mode)
    res = res_container.get()
    if isinstance(res, Exception):
        raise res
    return res

def idawrite(f):
    """
    decorator for marking a function as modifying the IDB.
    schedules a request to be made in the main IDA loop to avoid IDB corruption.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__ # type: ignore
        return sync_wrapper(ff, idaapi.MFF_WRITE)
    return wrapper

def idaread(f):
    """
    decorator for marking a function as reading from the IDB.
    schedules a request to be made in the main IDA loop to avoid
      inconsistent results.
    MFF_READ constant via: http://www.openrce.org/forums/posts/1827
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        ff = functools.partial(f, *args, **kwargs)
        ff.__name__ = f.__name__ # type: ignore
        return sync_wrapper(ff, idaapi.MFF_READ)
    return wrapper

def is_window_active():
    """Returns whether IDA is currently active"""
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        return False

    app = QApplication.instance()
    if app is None:
        return False

    for widget in app.topLevelWidgets():
        if widget.isActiveWindow():
            return True
    return False

class Metadata(TypedDict):
    path: str
    module: str
    base: str
    size: str
    md5: str
    sha256: str
    crc32: str
    filesize: str

def get_image_size() -> int:
    try:
        # https://www.hex-rays.com/products/ida/support/sdkdoc/structidainfo.html
        info = idaapi.get_inf_structure() # type: ignore
        omin_ea = info.omin_ea
        omax_ea = info.omax_ea
    except AttributeError:
        import ida_ida
        omin_ea = ida_ida.inf_get_omin_ea()
        omax_ea = ida_ida.inf_get_omax_ea()
    # Bad heuristic for image size (bad if the relocations are the last section)
    image_size = omax_ea - omin_ea
    # Try to extract it from the PE header
    header = idautils.peutils_t().header()
    if header and header[:4] == b"PE\0\0":
        image_size = struct.unpack("<I", header[0x50:0x54])[0]
    return image_size

@jsonrpc
@idaread
def get_metadata() -> Metadata:
    """Get metadata about the current IDB"""
    # Fat Mach-O binaries can return a None hash:
    # https://github.com/mrexodia/ida-pro-mcp/issues/26
    def hash(f):
        try:
            return f().hex()
        except:
            return ""

    return Metadata(path=idaapi.get_input_file_path(),
                    module=idaapi.get_root_filename(),
                    base=hex(idaapi.get_imagebase()),
                    size=hex(get_image_size()),
                    md5=hash(ida_nalt.retrieve_input_file_md5),
                    sha256=hash(ida_nalt.retrieve_input_file_sha256),
                    crc32=hex(ida_nalt.retrieve_input_file_crc32()),
                    filesize=hex(ida_nalt.retrieve_input_file_size()))

def get_prototype(fn: ida_funcs.func_t) -> Optional[str]:
    try:
        prototype: ida_typeinf.tinfo_t = fn.get_prototype()
        if prototype is not None:
            return str(prototype)
        else:
            return None
    except AttributeError:
        try:
            return idc.get_type(fn.start_ea)
        except:
            tif = ida_typeinf.tinfo_t()
            if ida_nalt.get_tinfo(tif, fn.start_ea):
                return str(tif)
            return None
    except Exception as e:
        print(f"Error getting function prototype: {e}")
        return None

class Function(TypedDict):
    address: str
    name: str
    size: str

def parse_address(address: str | int) -> int:
    if isinstance(address, int):
        return address
    try:
        return int(address, 0)
    except ValueError:
        for ch in address:
            if ch not in "0123456789abcdefABCDEF":
                raise IDAError(f"Failed to parse address: {address}")
        raise IDAError(f"Failed to parse address (missing 0x prefix): {address}")

@overload
def get_function(address: int, *, raise_error: Literal[True]) -> Function: ...

@overload
def get_function(address: int) -> Function: ...

@overload
def get_function(address: int, *, raise_error: Literal[False]) -> Optional[Function]: ...

def get_function(address, *, raise_error=True):
    fn = idaapi.get_func(address)
    if fn is None:
        if raise_error:
            raise IDAError(f"No function found at address {hex(address)}")
        return None

    try:
        name = fn.get_name()
    except AttributeError:
        name = ida_funcs.get_func_name(fn.start_ea)

    return Function(address=hex(address), name=name, size=hex(fn.end_ea - fn.start_ea))

DEMANGLED_TO_EA = {}

def create_demangled_to_ea_map():
    for ea in idautils.Functions():
        # Get the function name and demangle it
        # MNG_NODEFINIT inhibits everything except the main name
        # where default demangling adds the function signature
        # and decorators (if any)
        demangled = idaapi.demangle_name(
            idc.get_name(ea, 0), idaapi.MNG_NODEFINIT)
        if demangled:
            DEMANGLED_TO_EA[demangled] = ea

def get_type_by_name(type_name: str) -> ida_typeinf.tinfo_t:
    # 8-bit integers
    if type_name in ('int8', '__int8', 'int8_t', 'char', 'signed char'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT8)
    elif type_name in ('uint8', '__uint8', 'uint8_t', 'unsigned char', 'byte', 'BYTE'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT8)

    # 16-bit integers
    elif type_name in ('int16', '__int16', 'int16_t', 'short', 'short int', 'signed short', 'signed short int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT16)
    elif type_name in ('uint16', '__uint16', 'uint16_t', 'unsigned short', 'unsigned short int', 'word', 'WORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT16)

    # 32-bit integers
    elif type_name in ('int32', '__int32', 'int32_t', 'int', 'signed int', 'long', 'long int', 'signed long', 'signed long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT32)
    elif type_name in ('uint32', '__uint32', 'uint32_t', 'unsigned int', 'unsigned long', 'unsigned long int', 'dword', 'DWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT32)

    # 64-bit integers
    elif type_name in ('int64', '__int64', 'int64_t', 'long long', 'long long int', 'signed long long', 'signed long long int'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT64)
    elif type_name in ('uint64', '__uint64', 'uint64_t', 'unsigned int64', 'unsigned long long', 'unsigned long long int', 'qword', 'QWORD'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT64)

    # 128-bit integers
    elif type_name in ('int128', '__int128', 'int128_t', '__int128_t'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_INT128)
    elif type_name in ('uint128', '__uint128', 'uint128_t', '__uint128_t', 'unsigned int128'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_UINT128)

    # Floating point types
    elif type_name in ('float', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_FLOAT)
    elif type_name in ('double', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_DOUBLE)
    elif type_name in ('long double', 'ldouble'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_LDOUBLE)

    # Boolean type
    elif type_name in ('bool', '_Bool', 'boolean'):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_BOOL)

    # Void type
    elif type_name in ('void', ):
        return ida_typeinf.tinfo_t(ida_typeinf.BTF_VOID)

    # If not a standard type, try to get a named type
    tif = ida_typeinf.tinfo_t()
    if tif.get_named_type(None, type_name, ida_typeinf.BTF_STRUCT):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_TYPEDEF):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_ENUM):
        return tif

    if tif.get_named_type(None, type_name, ida_typeinf.BTF_UNION):
        return tif

    if tif := ida_typeinf.tinfo_t(type_name):
        return tif

    raise IDAError(f"Unable to retrieve {type_name} type info object")

@jsonrpc
@idaread
def get_function_by_name(
    name: Annotated[str, "Name of the function to get"]
) -> Function:
    """Get a function by its name"""
    function_address = idaapi.get_name_ea(idaapi.BADADDR, name)
    if function_address == idaapi.BADADDR:
        # If map has not been created yet, create it
        if len(DEMANGLED_TO_EA) == 0:
            create_demangled_to_ea_map()
        # Try to find the function in the map, else raise an error
        if name in DEMANGLED_TO_EA:
            function_address = DEMANGLED_TO_EA[name]
        else:
            raise IDAError(f"No function found with name {name}")
    return get_function(function_address)

@jsonrpc
@idaread
def get_function_by_address(
    address: Annotated[str, "Address of the function to get"],
) -> Function:
    """Get a function by its address"""
    return get_function(parse_address(address))

@jsonrpc
@idaread
def get_current_address() -> str:
    """Get the address currently selected by the user"""
    return hex(idaapi.get_screen_ea())

@jsonrpc
@idaread
def get_current_function() -> Optional[Function]:
    """Get the function currently selected by the user"""
    return get_function(idaapi.get_screen_ea())

class ConvertedNumber(TypedDict):
    decimal: str
    hexadecimal: str
    bytes: str
    ascii: Optional[str]
    binary: str

@jsonrpc
def convert_number(
    text: Annotated[str, "Textual representation of the number to convert"],
    size: Annotated[Optional[int], "Size of the variable in bytes"],
) -> ConvertedNumber:
    """Convert a number (decimal, hexadecimal) to different representations"""
    try:
        value = int(text, 0)
    except ValueError:
        raise IDAError(f"Invalid number: {text}")

    # Estimate the size of the number
    if not size:
        size = 0
        n = abs(value)
        while n:
            size += 1
            n >>= 1
        size += 7
        size //= 8

    # Convert the number to bytes
    try:
        bytes = value.to_bytes(size, "little", signed=True)
    except OverflowError:
        raise IDAError(f"Number {text} is too big for {size} bytes")

    # Convert the bytes to ASCII
    ascii = ""
    for byte in bytes.rstrip(b"\x00"):
        if byte >= 32 and byte <= 126:
            ascii += chr(byte)
        else:
            ascii = None
            break

    return ConvertedNumber(
        decimal=str(value),
        hexadecimal=hex(value),
        bytes=bytes.hex(" "),
        ascii=ascii,
        binary=bin(value),
    )

T = TypeVar("T")

class Page(TypedDict, Generic[T]):
    data: list[T]
    next_offset: Optional[int]

def paginate(data: list[T], offset: int, count: int) -> Page[T]:
    if count == 0:
        count = len(data)
    next_offset = offset + count
    if next_offset >= len(data):
        next_offset = None
    return {
        "data": data[offset:offset + count],
        "next_offset": next_offset,
    }

def pattern_filter(data: list[T], pattern: str, key: str) -> list[T]:
    if not pattern:
        return data

    # Check if pattern is a regex (wrapped in /.../)
    if pattern.startswith('/') and pattern.endswith('/') and len(pattern) > 2:
        import re
        regex_pattern = pattern[1:-1]  # Remove the / delimiters
        try:
            compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
            def matches(item) -> bool:
                return compiled_regex.search(item[key]) is not None
        except re.error:
            # If regex is invalid, fall back to literal matching
            def matches(item) -> bool:
                return pattern.lower() in item[key].lower()
    else:
        # Case-insensitive substring matching
        def matches(item) -> bool:
            return pattern.lower() in item[key].lower()
    
    return list(filter(matches, data))

@jsonrpc
@idaread
def list_functions(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of functions to list (100 is a good default, 0 means remainder)"],
    filter: Annotated[str, "Filter by function name (optional, empty string for no filter). Use plain text for case-insensitive substring match (e.g., 'malloc'), or wrap in /slashes/ for regex (e.g., '/^sub_[0-9A-F]+$/' for IDA default names)"] = "",
) -> Page[Function]:
    """List all functions in the database (paginated, optional filter)"""
    functions = [get_function(address) for address in idautils.Functions()]
    functions = pattern_filter(functions, filter, "name")
    return paginate(functions, offset, count)

class Global(TypedDict):
    address: str
    name: str

@jsonrpc
@idaread
def list_globals(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of globals to list (100 is a good default, 0 means remainder)"],
    filter: Annotated[str, "Filter by global name (optional, empty string for no filter). Use plain text for case-insensitive substring match (e.g., 'config'), or wrap in /slashes/ for regex (e.g., '/^g_.*_ptr$/' for globals ending with _ptr)"] = "",
) -> Page[Global]:
    """List globals in the database (paginated, optional filter)"""
    globals: list[Global] = []
    for addr, name in idautils.Names():
        # Skip functions and none
        if not idaapi.get_func(addr) or name is None:
            globals += [Global(address=hex(addr), name=name)]

    globals = pattern_filter(globals, filter, "name")
    return paginate(globals, offset, count)

class Import(TypedDict):
    address: str
    imported_name: str
    module: str

@jsonrpc
@idaread
def list_imports(
        offset: Annotated[int, "Offset to start listing from (start at 0)"],
        count: Annotated[int, "Number of imports to list (100 is a good default, 0 means remainder)"],
        filter: Annotated[str, "Filter by imported symbol name (optional, empty string for no filter). Use plain text for case-insensitive substring match (e.g., 'socket'), or wrap in /slashes/ for regex (e.g., '/Create.*W$/' for Unicode Create functions)"] = "",
) -> Page[Import]:
    """ List all imported symbols with their name and module (paginated, optional filter) """
    nimps = ida_nalt.get_import_module_qty()

    rv = []
    for i in range(nimps):
        module_name = ida_nalt.get_import_module_name(i)
        if not module_name:
            module_name = "<unnamed>"

        def imp_cb(ea, symbol_name, ordinal, acc):
            if not symbol_name:
                symbol_name = f"#{ordinal}"

            acc += [Import(address=hex(ea), imported_name=symbol_name, module=module_name)]

            return True

        imp_cb_w_context = lambda ea, symbol_name, ordinal: imp_cb(ea, symbol_name, ordinal, rv)
        ida_nalt.enum_import_names(i, imp_cb_w_context)

    rv = pattern_filter(rv, filter, "imported_name")
    return paginate(rv, offset, count)

class String(TypedDict):
    address: str
    length: int
    string: str

@jsonrpc
@idaread
def list_strings(
    offset: Annotated[int, "Offset to start listing from (start at 0)"],
    count: Annotated[int, "Number of strings to list (100 is a good default, 0 means remainder)"],
    filter: Annotated[str, "Filter by string content (optional, empty string for no filter). Use plain text for case-insensitive substring match (e.g., 'error'), or wrap in /slashes/ for regex (e.g., '/https?://.*/' for URLs)"] = "",
) -> Page[String]:
    """List strings in the database (paginated, optional filter)"""
    strings: list[String] = []
    for item in idautils.Strings():
        if item is None:
            continue
        try:
            string = str(item)
            if string:
                strings += [
                    String(address=hex(item.ea), length=item.length, string=string),
                ]
        except:
            continue
    strings = pattern_filter(strings, filter, "string")
    return paginate(strings, offset, count)

@jsonrpc
@idaread
def list_local_types():
    """List all Local types in the database"""
    error = ida_hexrays.hexrays_failure_t()
    locals = []
    idati = ida_typeinf.get_idati()
    type_count = ida_typeinf.get_ordinal_limit(idati)
    for ordinal in range(1, type_count):
        try:
            tif = ida_typeinf.tinfo_t()
            if tif.get_numbered_type(idati, ordinal):
                type_name = tif.get_type_name()
                if not type_name:
                    type_name = f"<Anonymous Type #{ordinal}>"
                locals.append(f"\nType #{ordinal}: {type_name}")
                if tif.is_udt():
                    c_decl_flags = (ida_typeinf.PRTYPE_MULTI | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI | ida_typeinf.PRTYPE_DEF | ida_typeinf.PRTYPE_METHODS | ida_typeinf.PRTYPE_OFFSETS)
                    c_decl_output = tif._print(None, c_decl_flags)
                    if c_decl_output:
                        locals.append(f"  C declaration:\n{c_decl_output}")
                else:
                    simple_decl = tif._print(None, ida_typeinf.PRTYPE_1LINE | ida_typeinf.PRTYPE_TYPE | ida_typeinf.PRTYPE_SEMI)
                    if simple_decl:
                        locals.append(f"  Simple declaration:\n{simple_decl}")
            else:
                message = f"\nType #{ordinal}: Failed to retrieve information."
                if error.str:
                    message += f": {error.str}"
                if error.errea != idaapi.BADADDR:
                    message += f"from (address: {hex(error.errea)})"
                raise IDAError(message)
        except:
            continue
    return locals

def decompile_checked(address: int) -> ida_hexrays.cfunc_t:
    if not ida_hexrays.init_hexrays_plugin():
        raise IDAError("Hex-Rays decompiler is not available")
    error = ida_hexrays.hexrays_failure_t()
    cfunc = ida_hexrays.decompile_func(address, error, ida_hexrays.DECOMP_WARNINGS)
    if not cfunc:
        if error.code == ida_hexrays.MERR_LICENSE:
            raise IDAError("Decompiler license is not available. Use `disassemble_function` to get the assembly code instead.")

        message = f"Decompilation failed at {hex(address)}"
        if error.str:
            message += f": {error.str}"
        if error.errea != idaapi.BADADDR:
            message += f" (address: {hex(error.errea)})"
        raise IDAError(message)
    return cfunc # type: ignore (this is a SWIG issue)

@jsonrpc
@idaread
def decompile_function(
    address: Annotated[str, "Address of the function to decompile"],
) -> str:
    """Decompile a function at the given address"""
    start = parse_address(address)
    cfunc = decompile_checked(start)
    if is_window_active():
        ida_hexrays.open_pseudocode(start, ida_hexrays.OPF_REUSE)
    sv = cfunc.get_pseudocode()
    pseudocode = ""
    for i, sl in enumerate(sv):
        sl: ida_kernwin.simpleline_t
        item = ida_hexrays.ctree_item_t()
        addr = None if i > 0 else cfunc.entry_ea
        if cfunc.get_line_item(sl.line, 0, False, None, item, None): # type: ignore (IDA SDK type hint wrong)
            dstr: str | None = item.dstr()
            if dstr:
                ds = dstr.split(": ")
                if len(ds) == 2:
                    try:
                        addr = int(ds[0], 16)
                    except ValueError:
                        pass
        line = ida_lines.tag_remove(sl.line)
        if len(pseudocode) > 0:
            pseudocode += "\n"
        if not addr:
            pseudocode += f"/* line: {i} */ {line}"
        else:
            pseudocode += f"/* line: {i}, address: {hex(addr)} */ {line}"

    return pseudocode

class DisassemblyLine(TypedDict):
    segment: NotRequired[str]
    address: str
    label: NotRequired[str]
    instruction: str
    comments: NotRequired[list[str]]

class Argument(TypedDict):
    name: str
    type: str

class StackFrameVariable(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class DisassemblyFunction(TypedDict):
    name: str
    start_ea: str
    return_type: NotRequired[str]
    arguments: NotRequired[list[Argument]]
    stack_frame: list[StackFrameVariable]
    lines: list[DisassemblyLine]

@jsonrpc
@idaread
def disassemble_function(
    start_address: Annotated[str, "Address of the function to disassemble"],
) -> DisassemblyFunction:
    """Get assembly code for a function"""
    start = parse_address(start_address)
    func: ida_funcs.func_t = idaapi.get_func(start)
    if not func:
        raise IDAError(f"No function found containing address {start_address}")
    if is_window_active():
        ida_kernwin.jumpto(start)

    lines = []
    for address in ida_funcs.func_item_iterator_t(func):
        seg = idaapi.getseg(address)
        segment = idaapi.get_segm_name(seg) if seg else None

        label = idc.get_name(address, 0)
        if label and label == func.name and address == func.start_ea:
            label = None
        if label == "":
            label = None

        comments = []
        if comment := idaapi.get_cmt(address, False):
            comments += [comment]
        if comment := idaapi.get_cmt(address, True):
            comments += [comment]

        raw_instruction = idaapi.generate_disasm_line(address, 0)
        tls = ida_kernwin.tagged_line_sections_t()
        if raw_instruction:
            ida_kernwin.parse_tagged_line_sections(tls, raw_instruction)
        insn_section = tls.first(ida_lines.COLOR_INSN) if raw_instruction else None

        operands = []
        if raw_instruction:
            for op_tag in range(ida_lines.COLOR_OPND1, ida_lines.COLOR_OPND8 + 1):
                op_n = tls.first(op_tag)
                if not op_n:
                    break

                try:
                    op: str = op_n.substr(raw_instruction)
                    op_str = ida_lines.tag_remove(op)
                except Exception:
                    # Be defensive against malformed tagged lines
                    continue

                # Do a lot of work to add address comments for symbols
                try:
                    for idx in range(len(op) - 2):
                        if op[idx] != idaapi.COLOR_ON:
                            continue

                        idx += 1
                        if ord(op[idx]) != idaapi.COLOR_ADDR:
                            continue

                        idx += 1
                        addr_string = op[idx:idx + idaapi.COLOR_ADDR_SIZE]
                        idx += idaapi.COLOR_ADDR_SIZE

                        addr = int(addr_string, 16)

                        # Find the next color and slice until there
                        symbol = op[idx:op.find(idaapi.COLOR_OFF, idx)]

                        if symbol == '':
                            # We couldn't figure out the symbol, so use the whole op_str
                            symbol = op_str

                        comments += [f"{symbol}={addr:#x}"]

                        # print its value if its type is available
                        try:
                            value = get_global_variable_value_internal(addr)
                            comments += [f"*{symbol}={value}"]
                        except Exception:
                            pass
                except Exception:
                    # Be resilient to any color parsing issues
                    pass

                operands += [op_str]

        # Build instruction string safely
        if insn_section and raw_instruction:
            try:
                mnem = ida_lines.tag_remove(insn_section.substr(raw_instruction))
                instruction = f"{mnem} {', '.join(operands)}".rstrip()
            except Exception:
                instruction = ida_lines.tag_remove(raw_instruction)
        else:
            instruction = ida_lines.tag_remove(raw_instruction) if raw_instruction else "db ?"

        line = DisassemblyLine(
            address=f"{address:#x}",
            instruction=instruction,
        )

        if len(comments) > 0:
            line.update(comments=comments)

        if segment:
            line.update(segment=segment)

        if label:
            line.update(label=label)

        lines += [line]

    prototype = func.get_prototype()
    arguments: list[Argument] = [Argument(name=arg.name, type=f"{arg.type}") for arg in prototype.iter_func()] if prototype else None

    disassembly_function = DisassemblyFunction(
        name=func.name,
        start_ea=f"{func.start_ea:#x}",
        stack_frame=get_stack_frame_variables_internal(func.start_ea, False),
        lines=lines
    )

    if prototype:
        disassembly_function.update(return_type=f"{prototype.get_rettype()}")

    if arguments:
        disassembly_function.update(arguments=arguments)

    return disassembly_function

@jsonrpc
@idaread
def disassemble_address(
    start_address: Annotated[str, "Starting address to disassemble from"],
    instruction_count: Annotated[int, "Number of instructions to disassemble"],
) -> list[DisassemblyLine]:
    """Disassemble a fixed number of instructions starting at the given address"""
    def _render_data_line(ea: int) -> tuple[str, int]:
        """Render a safe data representation at ea and return (text, size)."""
        try:
            # Try string literal first
            if ida_bytes.is_strlit(ea):
                s = idaapi.get_strlit_contents(ea, -1, 0)
                if isinstance(s, (bytes, bytearray)):
                    try:
                        text = s.decode("utf-8", errors="replace")
                    except Exception:
                        text = s.decode(errors="replace")
                else:
                    text = str(s)
                size = ida_bytes.get_item_size(ea) or len(s) or 1
                return (f"string \"{text}\"", size)
        except Exception:
            pass

        # Fallback to raw bytes
        size = ida_bytes.get_item_size(ea)
        if not size or size <= 0:
            size = 1
        try:
            raw = ida_bytes.get_bytes(ea, min(size, 16)) or b""
        except Exception:
            raw = b""
        bytes_repr = ", ".join(f"0x{b:02X}" for b in raw)
        if not bytes_repr:
            bytes_repr = "?"
        # Try to pick an assembler-like directive based on size when aligned
        directive = "db"
        if size in (2, 4, 8):
            # Check alignment heuristically
            if (ea % size) == 0:
                directive = {2: "dw", 4: "dd", 8: "dq"}[size]
                try:
                    val = {
                        2: ida_bytes.get_word,
                        4: ida_bytes.get_dword,
                        8: ida_bytes.get_qword,
                    }[size](ea)
                    return (f"{directive} 0x{val:X}", size)
                except Exception:
                    pass
        return (f"{directive} {bytes_repr}", size)

    current_ea = parse_address(start_address)
    max_ea = ida_ida.inf_get_max_ea()

    lines: list[DisassemblyLine] = []
    for _ in range(instruction_count):
        # Decode instruction; stop if decoding fails
        insn = idaapi.insn_t()
        decoded = idaapi.decode_insn(insn, current_ea)

        seg = idaapi.getseg(current_ea)
        segment = idaapi.get_segm_name(seg) if seg else None

        label = idc.get_name(current_ea, 0)
        if label == "":
            label = None

        comments = []
        if comment := idaapi.get_cmt(current_ea, False):
            comments += [comment]
        if comment := idaapi.get_cmt(current_ea, True):
            comments += [comment]

        instruction: str
        advanced_by: Optional[int] = None

        if not decoded:
            # Not an instruction (likely data). Render a safe data line and advance by item size.
            instruction, advanced_by = _render_data_line(current_ea)
        else:
            raw_instruction = idaapi.generate_disasm_line(current_ea, 0)
            tls = ida_kernwin.tagged_line_sections_t()
            if raw_instruction:
                ida_kernwin.parse_tagged_line_sections(tls, raw_instruction)
            insn_section = tls.first(ida_lines.COLOR_INSN) if raw_instruction else None

            operands = []
            if raw_instruction:
                for op_tag in range(ida_lines.COLOR_OPND1, ida_lines.COLOR_OPND8 + 1):
                    op_n = tls.first(op_tag)
                    if not op_n:
                        break

                    try:
                        op: str = op_n.substr(raw_instruction)
                        op_str = ida_lines.tag_remove(op)
                    except Exception:
                        # Defensive: if substr fails, skip this operand
                        continue

                    # Add address comments for operands when possible
                    try:
                        for idx in range(len(op) - 2):
                            if op[idx] != idaapi.COLOR_ON:
                                continue

                            idx += 1
                            if ord(op[idx]) != idaapi.COLOR_ADDR:
                                continue

                            idx += 1
                            addr_string = op[idx:idx + idaapi.COLOR_ADDR_SIZE]
                            idx += idaapi.COLOR_ADDR_SIZE

                            addr = int(addr_string, 16)

                            # Find the next color and slice until there
                            symbol = op[idx:op.find(idaapi.COLOR_OFF, idx)]

                            if symbol == '':
                                symbol = op_str

                            comments += [f"{symbol}={addr:#x}"]
                    except Exception:
                        # Be resilient to any color parsing issues
                        pass

                    operands += [op_str]

            if insn_section and raw_instruction:
                try:
                    mnem = ida_lines.tag_remove(insn_section.substr(raw_instruction))
                    instruction = f"{mnem} {', '.join(operands)}".rstrip()
                except Exception:
                    # Fallback to raw disassembly string if anything goes wrong
                    instruction = ida_lines.tag_remove(raw_instruction)
            else:
                # No instruction section (e.g., data or unparsable line); use raw disasm or data rendering
                if raw_instruction:
                    instruction = ida_lines.tag_remove(raw_instruction)
                else:
                    instruction, advanced_by = _render_data_line(current_ea)

        line = DisassemblyLine(
            address=f"{current_ea:#x}",
            instruction=instruction,
        )
        if len(comments) > 0:
            line.update(comments=comments)
        if segment:
            line.update(segment=segment)
        if label:
            line.update(label=label)

        lines += [line]

        # Advance to next address
        if advanced_by is not None:
            next_ea = current_ea + max(advanced_by, 1)
        else:
            next_ea = idc.next_head(current_ea, max_ea)
            if next_ea == idaapi.BADADDR or next_ea <= current_ea:
                break
        current_ea = next_ea

    return lines

class Xref(TypedDict):
    address: str
    type: str
    function: Optional[Function]

@jsonrpc
@idaread
def get_xrefs_to(
    address: Annotated[str, "Address to get cross references to"],
) -> list[Xref]:
    """Get all cross references to the given address"""
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(parse_address(address)): # type: ignore (IDA SDK type hints are incorrect)
        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs

@jsonrpc
@idaread
def get_xrefs_to_field(
    struct_name: Annotated[str, "Name of the struct (type) containing the field"],
    field_name: Annotated[str, "Name of the field (member) to get xrefs to"],
) -> list[Xref]:
    """Get all cross references to a named struct field (member)"""

    # Get the type library
    til = ida_typeinf.get_idati()
    if not til:
        raise IDAError("Failed to retrieve type library.")

    # Get the structure type info
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(til, struct_name, ida_typeinf.BTF_STRUCT, True, False):
        print(f"Structure '{struct_name}' not found.")
        return []

    # Get The field index
    idx = ida_typeinf.get_udm_by_fullname(None, struct_name + '.' + field_name) # type: ignore (IDA SDK type hints are incorrect)
    if idx == -1:
        print(f"Field '{field_name}' not found in structure '{struct_name}'.")
        return []

    # Get the type identifier
    tid = tif.get_udm_tid(idx)
    if tid == ida_idaapi.BADADDR:
        raise IDAError(f"Unable to get tid for structure '{struct_name}' and field '{field_name}'.")

    # Get xrefs to the tid
    xrefs = []
    xref: ida_xref.xrefblk_t
    for xref in idautils.XrefsTo(tid): # type: ignore (IDA SDK type hints are incorrect)
        xrefs += [
            Xref(address=hex(xref.frm),
                 type="code" if xref.iscode else "data",
                 function=get_function(xref.frm, raise_error=False))
        ]
    return xrefs

@jsonrpc
@idaread
def get_callees(
    function_address: Annotated[str, "Address of the function to get callee functions"],
) -> list[dict[str, str]]:
    """Get all the functions called (callees) by the function at function_address"""
    func_start = parse_address(function_address)
    func = idaapi.get_func(func_start)
    if not func:
        raise IDAError(f"No function found containing address {function_address}")
    func_end = idc.find_func_end(func_start)
    callees: list[dict[str, str]] = []
    current_ea = func_start
    while current_ea < func_end:
        insn = idaapi.insn_t()
        idaapi.decode_insn(insn, current_ea)
        if insn.itype in [idaapi.NN_call, idaapi.NN_callfi, idaapi.NN_callni]:
            target = idc.get_operand_value(current_ea, 0)
            target_type = idc.get_operand_type(current_ea, 0)
            # check if it's a direct call - avoid getting the indirect call offset
            if target_type in [idaapi.o_mem, idaapi.o_near, idaapi.o_far]:
                # in here, we do not use get_function because the target can be external function.
                # but, we should mark the target as internal/external function.
                func_type = (
                    "internal" if idaapi.get_func(target) is not None else "external"
                )
                func_name = idc.get_name(target)
                if func_name is not None:
                    callees.append(
                        {"address": hex(target), "name": func_name, "type": func_type}
                    )
        current_ea = idc.next_head(current_ea, func_end)

    # deduplicate callees
    unique_callee_tuples = {tuple(callee.items()) for callee in callees}
    unique_callees = [dict(callee) for callee in unique_callee_tuples]
    return unique_callees  # type: ignore

@jsonrpc
@idaread
def get_callers(
    function_address: Annotated[str, "Address of the function to get callers"],
) -> list[Function]:
    """Get all callers of the given address"""
    callers = {}
    for caller_address in idautils.CodeRefsTo(parse_address(function_address), 0):
        # validate the xref address is a function
        func = get_function(caller_address, raise_error=False)
        if not func:
            continue
        # load the instruction at the xref address
        insn = idaapi.insn_t()
        idaapi.decode_insn(insn, caller_address)
        # check the instruction is a call
        if insn.itype not in [idaapi.NN_call, idaapi.NN_callfi, idaapi.NN_callni]:
            continue
        # deduplicate callers by address
        callers[func["address"]] = func

    return list(callers.values())

@jsonrpc
@idaread
def get_entry_points() -> list[Function]:
    """Get all entry points in the database"""
    result = []
    for i in range(ida_entry.get_entry_qty()):
        ordinal = ida_entry.get_entry_ordinal(i)
        address = ida_entry.get_entry(ordinal)
        func = get_function(address, raise_error=False)
        if func is not None:
            result.append(func)
    return result

@jsonrpc
@idawrite
def set_comment(
    address: Annotated[str, "Address in the function to set the comment for"],
    comment: Annotated[str, "Comment text"],
):
    """Set a comment for a given address in the function disassembly and pseudocode"""
    ea = parse_address(address)

    if not idaapi.set_cmt(ea, comment, False):
        raise IDAError(f"Failed to set disassembly comment at {hex(ea)}")

    if not ida_hexrays.init_hexrays_plugin():
        return

    # Reference: https://cyber.wtf/2019/03/22/using-ida-python-to-analyze-trickbot/
    # Check if the address corresponds to a line
    try:
        cfunc = decompile_checked(ea)
    except IDAError:
        # Skip decompiler comment if decompilation fails
        return

    # Special case for function entry comments
    if ea == cfunc.entry_ea:
        idc.set_func_cmt(ea, comment, True)
        cfunc.refresh_func_ctext()
        return

    eamap = cfunc.get_eamap()
    if ea not in eamap:
        print(f"Failed to set decompiler comment at {hex(ea)}")
        return
    nearest_ea = eamap[ea][0].ea

    # Remove existing orphan comments
    if cfunc.has_orphan_cmts():
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()

    # Set the comment by trying all possible item types
    tl = idaapi.treeloc_t()
    tl.ea = nearest_ea
    for itp in range(idaapi.ITP_SEMI, idaapi.ITP_COLON):
        tl.itp = itp
        cfunc.set_user_cmt(tl, comment)
        cfunc.save_user_cmts()
        cfunc.refresh_func_ctext()
        if not cfunc.has_orphan_cmts():
            return
        cfunc.del_orphan_cmts()
        cfunc.save_user_cmts()
    print(f"Failed to set decompiler comment at {hex(ea)}")

def refresh_decompiler_widget():
    widget = ida_kernwin.get_current_widget()
    if widget is not None:
        vu = ida_hexrays.get_widget_vdui(widget)
        if vu is not None:
            vu.refresh_ctext()

def refresh_decompiler_ctext(function_address: int):
    error = ida_hexrays.hexrays_failure_t()
    cfunc: ida_hexrays.cfunc_t = ida_hexrays.decompile_func(function_address, error, ida_hexrays.DECOMP_WARNINGS)
    if cfunc:
        cfunc.refresh_func_ctext()

@jsonrpc
@idawrite
def rename_local_variable(
    function_address: Annotated[str, "Address of the function containing the variable"],
    old_name: Annotated[str, "Current name of the variable"],
    new_name: Annotated[str, "New name for the variable (empty for a default name)"],
):
    """Rename a local variable in a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, old_name, new_name):
        raise IDAError(f"Failed to rename local variable {old_name} in function {hex(func.start_ea)}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idawrite
def rename_global_variable(
    old_name: Annotated[str, "Current name of the global variable"],
    new_name: Annotated[str, "New name for the global variable (empty for a default name)"],
):
    """Rename a global variable"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, old_name)
    if not idaapi.set_name(ea, new_name):
        raise IDAError(f"Failed to rename global variable {old_name} to {new_name}")
    refresh_decompiler_ctext(ea)

@jsonrpc
@idawrite
def set_global_variable_type(
    variable_name: Annotated[str, "Name of the global variable"],
    new_type: Annotated[str, "New type for the variable"],
):
    """Set a global variable's type"""
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    tif = get_type_by_name(new_type)
    if not tif:
        raise IDAError(f"Parsed declaration is not a variable type")
    if not ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.PT_SIL):
        raise IDAError(f"Failed to apply type")

def patch_address_assemble(
    ea: int,
    assemble: str,
) -> int:
    """Patch Address Assemble"""
    (check_assemble, bytes_to_patch) = idautils.Assemble(ea, assemble)
    if check_assemble == False:
        raise IDAError(f"Failed to assemble instruction: {assemble}")
    try:
        ida_bytes.patch_bytes(ea, bytes_to_patch)
    except:
        raise IDAError(f"Failed to patch bytes at address {hex(ea)}")

    return len(bytes_to_patch)

@jsonrpc
@idawrite
def patch_address_assembles(
    address: Annotated[str, "Starting Address to apply patch"],
    instructions: Annotated[str, "Assembly instructions separated by ';' (e.g., 'nop; mov eax, 1; ret;')"],
) -> str:
    ea = parse_address(address)
    assembles = instructions.split(";")
    for assemble in assembles:
        assemble = assemble.strip()
        try:
            patch_bytes_len = patch_address_assemble(ea, assemble)
        except IDAError as e:
            raise IDAError(f"Failed to patch bytes at address {hex(ea)}: {e}")
        ea += patch_bytes_len
    return f"Patched {len(assembles)} instructions"

@jsonrpc
@idaread
def get_global_variable_value_by_name(variable_name: Annotated[str, "Name of the global variable"]) -> str:
    """
    Read a global variable's value (if known at compile-time)

    Prefer this function over the `data_read_*` functions.
    """
    ea = idaapi.get_name_ea(idaapi.BADADDR, variable_name)
    if ea == idaapi.BADADDR:
        raise IDAError(f"Global variable {variable_name} not found")

    return get_global_variable_value_internal(ea)

@jsonrpc
@idaread
def get_global_variable_value_at_address(address: Annotated[str, "Address of the global variable"]) -> str:
    """
    Read a global variable's value by its address (if known at compile-time)

    Prefer this function over the `data_read_*` functions.
    """
    ea = parse_address(address)
    return get_global_variable_value_internal(ea)

def get_global_variable_value_internal(ea: int) -> str:
     # Get the type information for the variable
     tif = ida_typeinf.tinfo_t()
     if not ida_nalt.get_tinfo(tif, ea):
         # No type info, maybe we can figure out its size by its name
         if not ida_bytes.has_any_name(ea):
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")

         size = ida_bytes.get_item_size(ea)
         if size == 0:
             raise IDAError(f"Failed to get type information for variable at {ea:#x}")
     else:
         # Determine the size of the variable
         size = tif.get_size()

     # Read the value based on the size
     if size == 0 and tif.is_array() and tif.get_array_element().is_decl_char():
         return_string = idaapi.get_strlit_contents(ea, -1, 0).decode("utf-8").strip()
         return f"\"{return_string}\""
     elif size == 1:
         return hex(ida_bytes.get_byte(ea))
     elif size == 2:
         return hex(ida_bytes.get_word(ea))
     elif size == 4:
         return hex(ida_bytes.get_dword(ea))
     elif size == 8:
         return hex(ida_bytes.get_qword(ea))
     else:
         # For other sizes, return the raw bytes
         return ' '.join(hex(x) for x in ida_bytes.get_bytes(ea, size))

@jsonrpc
@idawrite
def rename_function(
    function_address: Annotated[str, "Address of the function to rename"],
    new_name: Annotated[str, "New name for the function (empty for a default name)"],
):
    """Rename a function"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not idaapi.set_name(func.start_ea, new_name):
        raise IDAError(f"Failed to rename function {hex(func.start_ea)} to {new_name}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idawrite
def set_function_prototype(
    function_address: Annotated[str, "Address of the function"],
    prototype: Annotated[str, "New function prototype"],
):
    """Set a function's prototype"""
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    try:
        tif = ida_typeinf.tinfo_t(prototype, None, ida_typeinf.PT_SIL)
        if not tif.is_func():
            raise IDAError(f"Parsed declaration is not a function type")
        if not ida_typeinf.apply_tinfo(func.start_ea, tif, ida_typeinf.PT_SIL):
            raise IDAError(f"Failed to apply type")
        refresh_decompiler_ctext(func.start_ea)
    except Exception as e:
        raise IDAError(f"Failed to parse prototype string: {prototype}")

class my_modifier_t(ida_hexrays.user_lvar_modifier_t):
    def __init__(self, var_name: str, new_type: ida_typeinf.tinfo_t):
        ida_hexrays.user_lvar_modifier_t.__init__(self)
        self.var_name = var_name
        self.new_type = new_type

    def modify_lvars(self, lvinf):
        for lvar_saved in lvinf.lvvec:
            lvar_saved: ida_hexrays.lvar_saved_info_t
            if lvar_saved.name == self.var_name:
                lvar_saved.type = self.new_type
                return True
        return False

# NOTE: This is extremely hacky, but necessary to get errors out of IDA
def parse_decls_ctypes(decls: str, hti_flags: int) -> tuple[int, list[str]]:
    if sys.platform == "win32":
        import ctypes

        assert isinstance(decls, str), "decls must be a string"
        assert isinstance(hti_flags, int), "hti_flags must be an int"
        c_decls = decls.encode("utf-8")
        c_til = None
        ida_dll = ctypes.CDLL("ida")
        ida_dll.parse_decls.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        ida_dll.parse_decls.restype = ctypes.c_int

        messages: list[str] = []

        @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)
        def magic_printer(fmt: bytes, arg1: bytes):
            if fmt.count(b"%") == 1 and b"%s" in fmt:
                formatted = fmt.replace(b"%s", arg1)
                messages.append(formatted.decode("utf-8"))
                return len(formatted) + 1
            else:
                messages.append(f"unsupported magic_printer fmt: {repr(fmt)}")
                return 0

        errors = ida_dll.parse_decls(c_til, c_decls, magic_printer, hti_flags)
    else:
        # NOTE: The approach above could also work on other platforms, but it's
        # not been tested and there are differences in the vararg ABIs.
        errors = ida_typeinf.parse_decls(None, decls, False, hti_flags)
        messages = []
    return errors, messages

@jsonrpc
@idawrite
def declare_c_type(
    c_declaration: Annotated[str, "C declaration of the type. Examples include: typedef int foo_t; struct bar { int a; bool b; };"],
):
    """Create or update a local type from a C declaration"""
    # PT_SIL: Suppress warning dialogs (although it seems unnecessary here)
    # PT_EMPTY: Allow empty types (also unnecessary?)
    # PT_TYP: Print back status messages with struct tags
    flags = ida_typeinf.PT_SIL | ida_typeinf.PT_EMPTY | ida_typeinf.PT_TYP
    errors, messages = parse_decls_ctypes(c_declaration, flags)

    pretty_messages = "\n".join(messages)
    if errors > 0:
        raise IDAError(f"Failed to parse type:\n{c_declaration}\n\nErrors:\n{pretty_messages}")
    return f"success\n\nInfo:\n{pretty_messages}"

@jsonrpc
@idawrite
def set_local_variable_type(
    function_address: Annotated[str, "Address of the decompiled function containing the variable"],
    variable_name: Annotated[str, "Name of the variable"],
    new_type: Annotated[str, "New type for the variable"],
):
    """Set a local variable's type"""
    try:
        # Some versions of IDA don't support this constructor
        new_tif = ida_typeinf.tinfo_t(new_type, None, ida_typeinf.PT_SIL)
    except Exception:
        try:
            new_tif = ida_typeinf.tinfo_t()
            # parse_decl requires semicolon for the type
            ida_typeinf.parse_decl(new_tif, None, new_type + ";", ida_typeinf.PT_SIL) # type: ignore (IDA SDK type hints are incorrect)
        except Exception:
            raise IDAError(f"Failed to parse type: {new_type}")
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")
    if not ida_hexrays.rename_lvar(func.start_ea, variable_name, variable_name):
        raise IDAError(f"Failed to find local variable: {variable_name}")
    modifier = my_modifier_t(variable_name, new_tif)
    if not ida_hexrays.modify_user_lvars(func.start_ea, modifier):
        raise IDAError(f"Failed to modify local variable: {variable_name}")
    refresh_decompiler_ctext(func.start_ea)

@jsonrpc
@idaread
def get_stack_frame_variables(
        function_address: Annotated[str, "Address of the disassembled function to retrieve the stack frame variables"]
) -> list[StackFrameVariable]:
    """ Retrieve the stack frame variables for a given function """
    return get_stack_frame_variables_internal(parse_address(function_address), True)

def get_stack_frame_variables_internal(function_address: int, raise_error: bool) -> list[StackFrameVariable]:
    # TODO: IDA 8.3 does not support tif.get_type_by_tid
    if ida_major < 9:
        return []

    func = idaapi.get_func(function_address)
    if not func:
        if raise_error:
            raise IDAError(f"No function found at address {function_address}")
        return []

    tif = ida_typeinf.tinfo_t()
    if not tif.get_type_by_tid(func.frame) or not tif.is_udt():
        return []

    members: list[StackFrameVariable] = []
    udt = ida_typeinf.udt_type_data_t()
    tif.get_udt_details(udt)
    for udm in udt:
        if not udm.is_gap():
            name = udm.name
            offset = udm.offset // 8
            size = udm.size // 8
            type = str(udm.type)
            members.append(StackFrameVariable(
                name=name,
                offset=hex(offset),
                size=hex(size),
                type=type
            ))
    return members

class StructureMember(TypedDict):
    name: str
    offset: str
    size: str
    type: str

class StructureDefinition(TypedDict):
    name: str
    size: str
    members: list[StructureMember]

@jsonrpc
@idaread
def get_defined_structures() -> list[StructureDefinition]:
    """ Returns a list of all defined structures """

    rv = []
    limit = ida_typeinf.get_ordinal_limit()
    for ordinal in range(1, limit):
        tif = ida_typeinf.tinfo_t()
        tif.get_numbered_type(None, ordinal)
        if tif.is_udt():
            udt = ida_typeinf.udt_type_data_t()
            members = []
            if tif.get_udt_details(udt):
                members = [
                    StructureMember(name=x.name,
                                    offset=hex(x.offset // 8),
                                    size=hex(x.size // 8),
                                    type=str(x.type))
                    for _, x in enumerate(udt)
                ]

            rv += [StructureDefinition(name=tif.get_type_name(), # type: ignore (IDA SDK type hints are incorrect)
                                       size=hex(tif.get_size()),
                                       members=members)]

    return rv

@jsonrpc
@idaread
def analyze_struct_detailed(name: Annotated[str, "Name of the structure to analyze"]) -> dict:
    """Detailed analysis of a structure with all fields"""
    # Get tinfo object
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(None, name):
        raise IDAError(f"Structure '{name}' not found!")

    result = {
        "name": name,
        "type": str(tif._print()),
        "size": tif.get_size(),
        "is_udt": tif.is_udt()
    }

    if not tif.is_udt():
        result["error"] = "This is not a user-defined type!"
        return result

    # Get UDT (User Defined Type) details
    udt_data = ida_typeinf.udt_type_data_t()
    if not tif.get_udt_details(udt_data):
        result["error"] = "Failed to get structure details!"
        return result

    result["member_count"] = udt_data.size()
    result["is_union"] = udt_data.is_union
    result["udt_type"] = "Union" if udt_data.is_union else "Struct"

    # Output information about each field
    members = []
    for i, member in enumerate(udt_data):
        offset = member.begin() // 8  # Convert bits to bytes
        size = member.size // 8 if member.size > 0 else member.type.get_size()
        member_type = member.type._print()
        member_name = member.name

        member_info = {
            "index": i,
            "offset": f"0x{offset:08X}",
            "size": size,
            "type": member_type,
            "name": member_name,
            "is_nested_udt": member.type.is_udt()
        }

        # If this is a nested structure, show additional information
        if member.type.is_udt():
            member_info["nested_size"] = member.type.get_size()

        members.append(member_info)

    result["members"] = members
    result["total_size"] = tif.get_size()

    return result

@jsonrpc
@idaread
def get_struct_at_address(address: Annotated[str, "Address to analyze structure at"],
                         struct_name: Annotated[str, "Name of the structure"]) -> dict:
    """Get structure field values at a specific address"""
    addr = parse_address(address)

    # Get structure tinfo
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(None, struct_name):
        raise IDAError(f"Structure '{struct_name}' not found!")

    # Get structure details
    udt_data = ida_typeinf.udt_type_data_t()
    if not tif.get_udt_details(udt_data):
        raise IDAError("Failed to get structure details!")

    result = {
        "struct_name": struct_name,
        "address": f"0x{addr:X}",
        "members": []
    }

    for member in udt_data:
        offset = member.begin() // 8
        member_addr = addr + offset
        member_type = member.type._print()
        member_name = member.name
        member_size = member.type.get_size()

        # Try to get value based on size
        try:
            if member.type.is_ptr():
                # Pointer
                is_64bit = ida_ida.inf_is_64bit() if ida_major >= 9 else idaapi.get_inf_structure().is_64bit()
                if is_64bit:
                    value = idaapi.get_qword(member_addr)
                    value_str = f"0x{value:016X}"
                else:
                    value = idaapi.get_dword(member_addr)
                    value_str = f"0x{value:08X}"
            elif member_size == 1:
                value = idaapi.get_byte(member_addr)
                value_str = f"0x{value:02X} ({value})"
            elif member_size == 2:
                value = idaapi.get_word(member_addr)
                value_str = f"0x{value:04X} ({value})"
            elif member_size == 4:
                value = idaapi.get_dword(member_addr)
                value_str = f"0x{value:08X} ({value})"
            elif member_size == 8:
                value = idaapi.get_qword(member_addr)
                value_str = f"0x{value:016X} ({value})"
            else:
                # For large structures, read first few bytes
                bytes_data = []
                for i in range(min(member_size, 16)):
                    try:
                        byte_val = idaapi.get_byte(member_addr + i)
                        bytes_data.append(f"{byte_val:02X}")
                    except:
                        break
                value_str = f"[{' '.join(bytes_data)}{'...' if member_size > 16 else ''}]"
        except:
            value_str = "<failed to read>"

        member_info = {
            "offset": f"0x{offset:08X}",
            "type": member_type,
            "name": member_name,
            "value": value_str
        }

        result["members"].append(member_info)

    return result

@jsonrpc
@idaread
def get_struct_info_simple(name: Annotated[str, "Name of the structure"]) -> dict:
    """Simple function to get basic structure information"""
    tif = ida_typeinf.tinfo_t()
    if not tif.get_named_type(None, name):
        raise IDAError(f"Structure '{name}' not found!")

    info = {
        'name': name,
        'type': tif._print(),
        'size': tif.get_size(),
        'is_udt': tif.is_udt()
    }

    if tif.is_udt():
        udt_data = ida_typeinf.udt_type_data_t()
        if tif.get_udt_details(udt_data):
            info['member_count'] = udt_data.size()
            info['is_union'] = udt_data.is_union

            members = []
            for member in udt_data:
                members.append({
                    'name': member.name,
                    'type': member.type._print(),
                    'offset': member.begin() // 8,
                    'size': member.type.get_size()
                })
            info['members'] = members

    return info

@jsonrpc
@idaread
def search_structures(filter: Annotated[str, "Filter pattern to search for structures (case-insensitive)"]) -> list[dict]:
    """Search for structures by name pattern"""
    results = []
    limit = ida_typeinf.get_ordinal_limit()

    for ordinal in range(1, limit):
        tif = ida_typeinf.tinfo_t()
        if tif.get_numbered_type(None, ordinal):
            type_name: str = tif.get_type_name() # type: ignore (IDA SDK type hints are incorrect)
            if type_name and filter.lower() in type_name.lower():
                if tif.is_udt():
                    udt_data = ida_typeinf.udt_type_data_t()
                    member_count = 0
                    if tif.get_udt_details(udt_data):
                        member_count = udt_data.size()

                    results.append({
                        "name": type_name,
                        "size": tif.get_size(),
                        "member_count": member_count,
                        "is_union": udt_data.is_union if tif.get_udt_details(udt_data) else False,
                        "ordinal": ordinal
                    })

    return results

@jsonrpc
@idawrite
def rename_stack_frame_variable(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        old_name: Annotated[str, "Current name of the variable"],
        new_name: Annotated[str, "New name for the variable (empty for a default name)"]
):
    """ Change the name of a stack variable for an IDA function """
    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(old_name) # type: ignore (IDA SDK type hints are incorrect)
    if not udm:
        raise IDAError(f"{old_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{old_name} is a special frame member. Will not change the name.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{old_name} is an argument member. Will not change the name.")

    sval = ida_frame.soff_to_fpoff(func, offset)
    if not ida_frame.define_stkvar(func, new_name, sval, udm.type):
        raise IDAError("failed to rename stack frame variable")

@jsonrpc
@idawrite
def create_stack_frame_variable(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        offset: Annotated[str, "Offset of the stack frame variable"],
        variable_name: Annotated[str, "Name of the stack variable"],
        type_name: Annotated[str, "Type of the stack variable"]
):
    """ For a given function, create a stack variable at an offset and with a specific type """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    ea = parse_address(offset)

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    tif = get_type_by_name(type_name)
    if not ida_frame.define_stkvar(func, variable_name, ea, tif):
        raise IDAError("failed to define stack frame variable")

@jsonrpc
@idawrite
def set_stack_frame_variable_type(
        function_address: Annotated[str, "Address of the disassembled function to set the stack frame variables"],
        variable_name: Annotated[str, "Name of the stack variable"],
        type_name: Annotated[str, "Type of the stack variable"]
):
    """ For a given disassembled function, set the type of a stack variable """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name) # type: ignore (IDA SDK type hints are incorrect)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8

    tif = get_type_by_name(type_name)
    if not ida_frame.set_frame_member_type(func, offset, tif):
        raise IDAError("failed to set stack frame variable type")

@jsonrpc
@idawrite
def delete_stack_frame_variable(
        function_address: Annotated[str, "Address of the function to set the stack frame variables"],
        variable_name: Annotated[str, "Name of the stack variable"]
):
    """ Delete the named stack variable for a given function """

    func = idaapi.get_func(parse_address(function_address))
    if not func:
        raise IDAError(f"No function found at address {function_address}")

    frame_tif = ida_typeinf.tinfo_t()
    if not ida_frame.get_func_frame(frame_tif, func):
        raise IDAError("No frame returned.")

    idx, udm = frame_tif.get_udm(variable_name) # type: ignore (IDA SDK type hints are incorrect)
    if not udm:
        raise IDAError(f"{variable_name} not found.")

    tid = frame_tif.get_udm_tid(idx)
    if ida_frame.is_special_frame_member(tid):
        raise IDAError(f"{variable_name} is a special frame member. Will not delete.")

    udm = ida_typeinf.udm_t()
    frame_tif.get_udm_by_tid(udm, tid)
    offset = udm.offset // 8
    size = udm.size // 8
    if ida_frame.is_funcarg_off(func, offset):
        raise IDAError(f"{variable_name} is an argument member. Will not delete.")

    if not ida_frame.delete_frame_members(func, offset, offset+size):
        raise IDAError("failed to delete stack frame variable")

@jsonrpc
@idaread
def read_memory(
    address: Annotated[str, "Memory address to read from (hex or decimal)"],
    format: Annotated[str, "GDB-style format: [count][format][size]. Format: x(hex)/d(signed)/u(unsigned)/o(octal)/t(binary)/c(char)/s(string). Size: b(byte)/h(halfword-2)/w(word-4)/g(giant-8). Examples: 'xb'=1 hex byte, '16xb'=16 hex bytes, 'xw'=1 hex dword, 'dg'=1 signed qword, 's'=null-terminated string"],
) -> str:
    """
    Read memory at an address with GDB-style format specifiers.
    
    Format syntax: [count][format][size]
    - count (optional): number of units to read (default: 1)
    - format (required): x=hex, d=signed decimal, u=unsigned decimal, o=octal, t=binary, c=char, s=string
    - size (optional for numeric, not used for string): b=byte(1), h=halfword(2), w=word(4), g=giant(8). Default: w(4)
    
    Examples:
    - read_memory(addr, "xb") -> "0x42" (1 byte in hex)
    - read_memory(addr, "xw") -> "0x12345678" (1 dword in hex)
    - read_memory(addr, "xg") -> "0x123456789abcdef0" (1 qword in hex)
    - read_memory(addr, "16xb") -> "0x41 0x42 0x43..." (16 bytes in hex)
    - read_memory(addr, "dw") -> "305419896" (1 dword as signed decimal)
    - read_memory(addr, "s") -> "Hello World" (null-terminated string)
    """
    import re
    
    ea = parse_address(address)
    
    # Parse format string: [count][format][size]
    match = re.match(r'^(\d*)([xduotcs])([bhwg]?)$', format.lower())
    if not match:
        raise IDAError(f"Invalid format string: '{format}'. Expected [count][format][size] like 'xb', '16xb', 'dw', 's'")
    
    count_str, fmt, size = match.groups()
    count = int(count_str) if count_str else 1
    
    # Handle string format specially
    if fmt == 's':
        try:
            string_bytes = idaapi.get_strlit_contents(ea, -1, 0)
            if string_bytes is None:
                return ""
            return string_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading string: {str(e)}"
    
    # Determine size in bytes
    size_map = {'b': 1, 'h': 2, 'w': 4, 'g': 8}
    byte_size = size_map.get(size, 4)  # default to word (4 bytes)
    
    # Read the raw bytes
    try:
        raw_bytes = ida_bytes.get_bytes(ea, count * byte_size)
        if raw_bytes is None:
            raise IDAError(f"Failed to read {count * byte_size} bytes at {address}")
    except Exception as e:
        raise IDAError(f"Error reading memory: {str(e)}")
    
    # Parse bytes into values based on size
    values = []
    for i in range(count):
        offset = i * byte_size
        chunk = raw_bytes[offset:offset + byte_size]
        
        # Convert bytes to integer value
        if byte_size == 1:
            val = chunk[0]
        else:
            val = int.from_bytes(chunk, byteorder='little', signed=False)
        
        # Format based on format character
        if fmt == 'x':  # hexadecimal
            if byte_size == 1:
                values.append(f"0x{val:02x}")
            elif byte_size == 2:
                values.append(f"0x{val:04x}")
            elif byte_size == 4:
                values.append(f"0x{val:08x}")
            elif byte_size == 8:
                values.append(f"0x{val:016x}")
        elif fmt == 'd':  # signed decimal
            # Convert to signed
            max_val = 1 << (byte_size * 8)
            if val >= max_val // 2:
                val = val - max_val
            values.append(str(val))
        elif fmt == 'u':  # unsigned decimal
            values.append(str(val))
        elif fmt == 'o':  # octal
            values.append(f"0o{val:o}")
        elif fmt == 't':  # binary
            values.append(f"0b{val:0{byte_size*8}b}")
        elif fmt == 'c':  # character
            if 32 <= val <= 126:
                values.append(f"'{chr(val)}'")
            else:
                values.append(f"'\\x{val:02x}'")
    
    return ' '.join(values)

class RegisterValue(TypedDict):
    name: str
    value: str

class ThreadRegisters(TypedDict):
    thread_id: int
    registers: list[RegisterValue]

def dbg_ensure_running() -> "ida_idd.debugger_t":
    dbg = ida_idd.get_dbg()
    if not dbg:
        raise IDAError("Debugger not running")
    if ida_dbg.get_ip_val() is None:
        raise IDAError("Debugger not running")
    return dbg

@jsonrpc
@idaread
@unsafe
def dbg_get_registers() -> list[ThreadRegisters]:
    """Get all registers and their values. This function is only available when debugging."""
    result: list[ThreadRegisters] = []
    dbg = dbg_ensure_running()
    for thread_index in range(ida_dbg.get_thread_qty()):
        tid = ida_dbg.getn_thread(thread_index)
        regs = []
        regvals: ida_idd.regvals_t = ida_dbg.get_reg_vals(tid)
        for reg_index, rv in enumerate(regvals):
            rv: ida_idd.regval_t
            reg_info = dbg.regs(reg_index)

            # NOTE: Apparently this can fail under some circumstances
            try:
                reg_value = rv.pyval(reg_info.dtype)
            except ValueError:
                reg_value = ida_idaapi.BADADDR

            if isinstance(reg_value, int):
                reg_value = hex(reg_value)
            if isinstance(reg_value, bytes):
                reg_value = reg_value.hex(" ")
            else:
                reg_value = str(reg_value)
            regs.append({
                "name": reg_info.name,
                "value": reg_value,
            })
        result.append({
            "thread_id": tid,
            "registers": regs,
        })
    return result

@jsonrpc
@idaread
@unsafe
def dbg_get_call_stack() -> list[dict[str, str]]:
    """Get the current call stack."""
    callstack = []
    try:
        tid = ida_dbg.get_current_thread()
        trace = ida_idd.call_stack_t()

        if not ida_dbg.collect_stack_trace(tid, trace):
            return []
        for frame in trace:
            frame_info = {
                "address": hex(frame.callea),
            }
            try:
                module_info = ida_idd.modinfo_t()
                if ida_dbg.get_module_info(frame.callea, module_info):
                    frame_info["module"] = os.path.basename(module_info.name)
                else:
                    frame_info["module"] = "<unknown>"

                name = (
                    ida_name.get_nice_colored_name(
                        frame.callea,
                        ida_name.GNCN_NOCOLOR
                        | ida_name.GNCN_NOLABEL
                        | ida_name.GNCN_NOSEG
                        | ida_name.GNCN_PREFDBG,
                    )
                    or "<unnamed>"
                )
                frame_info["symbol"] = name

            except Exception as e:
                frame_info["module"] = "<error>"
                frame_info["symbol"] = str(e)

            callstack.append(frame_info)

    except Exception as e:
        pass
    return callstack

class Breakpoint(TypedDict):
    ea: str
    enabled: bool
    condition: Optional[str]

def list_breakpoints():
    breakpoints: list[Breakpoint] = []
    for i in range(ida_dbg.get_bpt_qty()):
        bpt = ida_dbg.bpt_t()
        if ida_dbg.getn_bpt(i, bpt):
            breakpoints.append(Breakpoint(
                ea=hex(bpt.ea),
                enabled=bpt.flags & ida_dbg.BPT_ENABLED,
                condition=str(bpt.condition) if bpt.condition else None,
            ))
    return breakpoints

@jsonrpc
@idaread
@unsafe
def dbg_list_breakpoints():
    """List all breakpoints in the program."""
    return list_breakpoints()

@jsonrpc
@idaread
@unsafe
def dbg_start_process():
    """Start the debugger, returns the current instruction pointer"""

    if len(list_breakpoints()) == 0:
        for i in range(ida_entry.get_entry_qty()):
            ordinal = ida_entry.get_entry_ordinal(i)
            address = ida_entry.get_entry(ordinal)
            if address != ida_idaapi.BADADDR:
                ida_dbg.add_bpt(address, 0, idaapi.BPT_SOFT)

    if idaapi.start_process("", "", "") == 1:
        ip = ida_dbg.get_ip_val()
        if ip is not None:
            return hex(ip)
    raise IDAError("Failed to start debugger (did the user configure the debugger manually one time?)")

@jsonrpc
@idaread
@unsafe
def dbg_exit_process():
    """Exit the debugger"""
    dbg_ensure_running()
    if idaapi.exit_process():
        return
    raise IDAError("Failed to exit debugger")

@jsonrpc
@idaread
@unsafe
def dbg_continue_process() -> str:
    """Continue the debugger, returns the current instruction pointer"""
    dbg_ensure_running()
    if idaapi.continue_process():
        ip = ida_dbg.get_ip_val()
        if ip is not None:
            return hex(ip)
    raise IDAError("Failed to continue debugger")

@jsonrpc
@idaread
@unsafe
def dbg_run_to(
    address: Annotated[str, "Run the debugger to the specified address"],
):
    """Run the debugger to the specified address"""
    dbg_ensure_running()
    ea = parse_address(address)
    if idaapi.run_to(ea):
        ip = ida_dbg.get_ip_val()
        if ip is not None:
            return hex(ip)
    raise IDAError(f"Failed to run to address {hex(ea)}")

@jsonrpc
@idaread
@unsafe
def dbg_set_breakpoint(
    address: Annotated[str, "Set a breakpoint at the specified address"],
):
    """Set a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.add_bpt(ea, 0, idaapi.BPT_SOFT):
        return f"Breakpoint set at {hex(ea)}"
    breakpoints = list_breakpoints()
    for bpt in breakpoints:
        if bpt["ea"] == hex(ea):
            return
    raise IDAError(f"Failed to set breakpoint at address {hex(ea)}")

@jsonrpc
@idaread
@unsafe
def dbg_step_into():
    """Step into the current instruction"""
    dbg_ensure_running()
    if idaapi.step_into():
        ip = ida_dbg.get_ip_val()
        if ip is not None:
            return hex(ip)
    raise IDAError("Failed to step into")

@jsonrpc
@idaread
@unsafe
def dbg_step_over():
    """Step over the current instruction"""
    dbg_ensure_running()
    if idaapi.step_over():
        ip = ida_dbg.get_ip_val()
        if ip is not None:
            return hex(ip)
    raise IDAError("Failed to step over")

@jsonrpc
@idaread
@unsafe
def dbg_delete_breakpoint(
    address: Annotated[str, "del a breakpoint at the specified address"],
):
    """Delete a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.del_bpt(ea):
        return
    raise IDAError(f"Failed to delete breakpoint at address {hex(ea)}")

@jsonrpc
@idaread
@unsafe
def dbg_enable_breakpoint(
    address: Annotated[str, "Enable or disable a breakpoint at the specified address"],
    enable: Annotated[bool, "Enable or disable a breakpoint"],
):
    """Enable or disable a breakpoint at the specified address"""
    ea = parse_address(address)
    if idaapi.enable_bpt(ea, enable):
        return
    raise IDAError(f"Failed to {'' if enable else 'disable '}breakpoint at address {hex(ea)}")

class MCP(idaapi.plugin_t):
    flags = idaapi.PLUGIN_KEEP
    comment = "MCP Plugin"
    help = "MCP"
    wanted_name = "MCP"
    wanted_hotkey = "Ctrl-Alt-M"

    def init(self):
        self.server = Server()
        hotkey = MCP.wanted_hotkey.replace("-", "+")
        if sys.platform == "darwin":
            hotkey = hotkey.replace("Alt", "Option")
        print(f"[MCP] Plugin loaded, use Edit -> Plugins -> MCP ({hotkey}) to toggle/configure the server")
        return idaapi.PLUGIN_KEEP

    def run(self, arg):
        """Handle plugin menu action - show config dialog or stop server"""
        if self.server.running:
            # Server is running - ask if user wants to stop it
            try:
                import ida_kernwin
                response = ida_kernwin.ask_yn(ida_kernwin.ASKBTN_YES, 
                    f"MCP server is running at http://{self.server.host}:{self.server.port}\n\nDo you want to stop it?")
                if response == ida_kernwin.ASKBTN_YES:
                    self.server.stop()
                    # ida_kernwin.info(f"MCP server stopped")
            except:
                # Fallback to console
                print(f"[MCP] Server is running at http://{self.server.host}:{self.server.port}")
                response = input("Stop the server? (y/n): ").strip().lower()
                if response == 'y':
                    self.server.stop()
        else:
            # Server is not running - show config dialog
            if self.server.configure():
                self.server.start()

    def term(self):
        self.server.stop()

def PLUGIN_ENTRY():
    return MCP()
