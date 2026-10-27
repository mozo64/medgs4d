import open3d as o3d
import argparse

def main(ply_file):
    mesh = o3d.io.read_triangle_mesh(ply_file)
    mesh.compute_vertex_normals()

    print(mesh)
    print("Vertices:", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a PLY mesh")
    parser.add_argument("mesh", help="Path to .ply file")
    args = parser.parse_args()
    main(args.mesh)
