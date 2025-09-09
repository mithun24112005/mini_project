// tb_alu.v
`timescale 1ns/1ps
module tb_alu;

    reg [31:0] a, b;
    reg [2:0] opcode;
    wire [31:0] result;

    reg [31:0] faulty_result;
    reg [2:0] fault_type; // 0 = no_fault, 1 = bitflip, 2 = opcode_fault

    integer i;
    integer bit_to_flip;

    // Instantiate ALU
    alu uut (
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result)
    );

    initial begin
        // Print CSV header once
        $display("a,b,opcode,faulty_result,fault_type");

        // Generate 25000 test cases
        for (i = 0; i < 5000; i = i + 1) begin
            // Random inuts
            a = $random;
            b = $random;
            opcode = $random % 5;  // 5 valid opcodes (ADD,SUB,AND,OR,XOR)

            #1; // wait for ALU to compute

            // Decide fault type
            case ($urandom_range(0,2))
                0: begin
                    // No fault
                    faulty_result = result;
                    fault_type = 0;
                end
                1: begin
                    // Bitflip fault
                    bit_to_flip = $urandom_range(0,31);
                    faulty_result = result ^ (32'h1 << bit_to_flip);
                    fault_type = 1;
                end
                2: begin
                    // Opcode fault (force XOR regardless of opcode)
                    faulty_result = (a ^ b);
                    fault_type = 2;
                end
            endcase

            // Print as CSV row to console
            $display("%0d,%0d,%0d,%0d,%0d", a, b, opcode, faulty_result, fault_type);
        end

        $finish;
    end
endmodule
